import pyclups
import numpy as np
from matplotlib import pyplot as pt
import scipy.linalg as lg

class OptimiserProperties:

	def __init__(self):
		self.b1 = 0.7
		self.b2 = 0.9
		self.alpha = 0.1
		self.MaxSteps =3000
		self.GradientCriteria = 1e-7
		self.ScoreCriteria =1e-7
		self.AlphaFactor = 5
		self.CurrentAlpha = 0.05
		self.gradMem = 0
		self.scoreMem = 0
		self.prevScore= 0
		self.deltaScore = 0
		

	def Start(self,verbose):
		self.Converged = (self.MaxSteps == 1)
		self.CurrentAlpha = self.alpha
		self.MinAlpha = self.CurrentAlpha/self.AlphaFactor
		self.MaxAlpha = self.CurrentAlpha*self.AlphaFactor
		self.AlphaTrigger = 0
		self.Verbose = verbose
	def Corrections(self,l):
		self.c1 = 1.0/(1.0 - pow(self.b1,l+1))
		self.c2 = 1.0/(1.0 - pow(self.b2,l+1))

	def _CheckConvergence(self,l,mse,grad):
		self.Converged=False
		convMem = 0.9
		earlyCorrector = 1.0/(1.0 - pow(convMem,l+1))
		
		gNorm = np.linalg.norm(grad/len(grad))
		self.gradMem = convMem * self.gradMem + (1.0 - convMem) * gNorm

		if l > 5 and self.gradMem * earlyCorrector < self.GradientCriteria:
			if self.Verbose:
				print(f"Convergence Criteria met: Gradient flat ({self.gradMem:.4}) at step {l}")
			self.Converged = True
		alphaCorrector = self.CurrentAlpha/self.alpha
		self.scoreMem = convMem * self.scoreMem + (1.0 - convMem) * mse
		scoreDelta = abs((mse - self.scoreMem)/self.scoreMem)
		self.deltaScore = convMem * self.deltaScore + (1.0 - convMem) * scoreDelta
		if l > 5 and self.deltaScore * earlyCorrector < self.ScoreCriteria:
			if self.Verbose:
				print(f"Convergence Criteria met: Function value stable at, {mse:.6} with mean change {100*self.deltaScore:.4}\%",l,alphaCorrector)
			self.Converged = True

		if (l >= self.MaxSteps):
			if self.Verbose:
				print(f"Convergence Criteria met: Took {l} steps, reached {mse:.6}")
			self.Converged = True

		## some extensions to the ADAM optimiser to make it take smaller steps when large oscillations
		if l > 5:
			if mse > self.prevScore:
				self.AlphaTrigger += 2
				if self.AlphaTrigger > 10:
					self.CurrentAlpha *= 0.9
					self.AlphaTrigger = 0
					if self.CurrentAlpha < self.MinAlpha:
						self.CurrentAlpha = self.MinAlpha
						self.MinAlpha = self.MinAlpha * 0.9
			else:
				min = -5
				self.AlphaTrigger = self.AlphaTrigger-1
				if self.AlphaTrigger < min:
					self.AlphaTrigger = 0
					self.CurrentAlpha *= 1.02
					if self.CurrentAlpha > self.MaxAlpha:
						self.CurrentAlpha = self.MaxAlpha
						self.MaxAlpha = self.MaxAlpha *1.1
		self.prevScore = mse
		# print("ALpha check",self.CurrentAlpha,self.MinAlpha,self.MaxAlpha)
class Predictor:
	trueFunc = None
	
	def __init__(self,kernel,constraint,basis,verbose=False):
		self.Kernel = kernel
		self.Constraints = constraint
		self.Basis = basis
		self.Verbose = verbose
		self.Optim = OptimiserProperties()

	def Predict(self,predictPoints,data,regulariser=None):
		self.Constraints.Validate(predictPoints)
		self.Regulariser = regulariser

		if regulariser is not None:
			self.Constraints.BulkUp(predictPoints)
			
			##prepare the regulariser!

		self._InitialiseComponents(predictPoints,data)
		if self.Constraints.IsConstant:
			if np.shape(self.PseudoInv)!=(0,0):

				corrector = self.PseudoInv@(self.Constraints.Vector() - self.Constraints.Matrix()@self.p_blups)				
				for i in range(len(predictPoints)):
					ai = self.a_blups[i] + corrector[i]* self.epsilon
					self.p_clups[i] = ai.T@data.X
			else:
				self.p_clups = self.p_blups

		else:
			self._Optimise(predictPoints)
			corrector = self.PseudoInv@(self.Constraints.Vector() - self.Constraints.Matrix()@self.p_blups)				
			for i in range(len(predictPoints)):
				ai = self.a_blups[i] + corrector[i] * self.epsilon
				self.p_clups[i] = ai.T@data.X

		if regulariser is not None:
			self.Constraints.Remove()
		return pyclups.Prediction(predictPoints,self.p_clups,self.p_blups,self.p_blps)

	def _InitialiseComponents(self,predictPoints,data):
		eX = np.array(data.Errors)
		if len(np.atleast_1d(eX)) == 1:
			dataVariance = np.ones(len(data.X)) * data.Errors**2
		else:
			if len(eX) != len(data.X):
				raise ValueError(f"Data length {len(data.X)} and error length {len(data.Errors)} are not concordant")
			dataVariance= eX**2

	
		dx = np.array(data.X).reshape(-1,1)
		self.Phi = np.zeros((self.Basis.maxOrder+1,len(data.T)))
		for i in range(len(data.T)):
			for m in range(self.Basis.maxOrder+1):
				self.Phi[m,i] = self.Basis(m,data.T[i])

		##scope for some fancy cholesky stuff here -- will do boring way first to get it working
		
		self.K = self.Kernel.Matrix(data.T,dataVariance)
		self.Kinv = np.linalg.inv(self.K)
		
		Minv = np.linalg.inv(self.Phi@self.Kinv@self.Phi.T)

		C = self.Kinv@self.Phi.T@Minv
		Delta = (np.eye(len(data.T)) - C@self.Phi).T

		self.ks = [np.zeros((0))]*len(predictPoints)
		self.a_blups = [np.zeros((0))]*len(predictPoints)
		self.p_blps = np.zeros((len(predictPoints),1))
		self.p_blups = np.zeros((len(predictPoints),1))
		self.p_clups = np.zeros((len(predictPoints),1))
		
		delta = self.Kinv@Delta@dx
		beta = dx.T@delta

		self.epsilon = 1.0/beta * delta
		D = self.Constraints.Matrix()
		self.Prefactor = 2.0/beta
		self.DDtInv = np.linalg.inv(D@D.T)
		self.PseudoInv = D.T @ self.DDtInv
		
		for i in range(len(predictPoints)):
			self.ks[i] = self.Kernel.Vector(data.T,predictPoints[i])
			a_blps = self.Kinv@self.ks[i]
			phi = np.array([self.Basis(j,predictPoints[i]) for j in range(self.Basis.maxOrder+1)]).reshape(-1,1)

			self.p_blps[i] = dx.T@a_blps
			self.a_blups[i] =  Delta.T@a_blps + C@phi
			self.p_blups[i] = dx.T@self.a_blups[i]

		self.Bp_blub = self.Constraints.Matrix() @ self.p_blups

		if self.Regulariser is not None:
			self.Binv = np.linalg.inv(D)
	# def _Step(self,l,alpha,grad):
	# 	#compute adam step rules
	# 	self.ms = self.optimiser_b1 * self.ms + (1.0 - self.optimiser_b1) * grad.reshape(self.ms.shape)
	# 	self.vs = self.optimiser_b2 * self.vs + (1.0 - self.optimiser_b2)*np.multiply(grad,grad).reshape(self.ms.shape)
	# 	c1 = 1.0/(1.0 - pow(self.optimiser_b1,l+1))
	# 	c2 = 1.0/(1.0 - pow(self.optimiser_b2,l+1))
	# 	# print("g=",grad.T)
	# 	# print("m=",self.ms.T)
	# 	step = -alpha*grad #np.divide(self.ms/c1, np.sqrt(self.vs/c2 + 1e-20))
	# 	print(self.ms[:5].T)
	# 	print(self.vs[:5].T)
	# 	print(-alpha *self.ms[1]/c1 / np.sqrt(self.vs[1]/c2))
	# 	print("STEP",step[:5].T)
	# 	# print(alpha,np.linalg.norm(step))
	# 	self.Constraints.Update(step)

	
	def _Optimise(self, predictPoints):

		#find a good initial position
		self.Constraints.InitialPosition(self.p_blups)
		self.Optim.Start(self.Verbose)
		#values for keeping track of convergence
		
		if self.Regulariser is not None:
			self.p_clups = self.Binv@self.Constraints.Vector()
		
		minScore = self._ComputeScore(predictPoints)
		mse = minScore
		self.Constraints.SavePosition()
		bc = self.Constraints.Vector()
		#begin the optimisation loop
		# currentAlpha = self.alpha
		# self.MinAlpha = currentAlpha/self.AlphaFactor
		# self.MaxAlpha = currentAlpha*self.AlphaFactor
		# self.AlphaTrigger = 0
		l = 0
		timeSinceDecrease = 0
		timeTrigger = 1000
		bestAngles= []
		while not self.Optim.Converged:
			#compute gradient
			c = self.Constraints.Vector()

			diff = self.Prefactor * self.DDtInv@(c - self.Bp_blub)
			if self.Regulariser is not None:
				self.p_clups = self.Binv@c

				diff += self.Binv.T @ self.Regulariser.dF(self.p_clups)
			# if (l+1) % 1 == 0:
			# 	# print()
			# 	# print(currentAlpha,mse,np.linalg.norm(grad))
			# 	print(c[:10].T)
			# 	pt.plot(predictPoints,bc)
			# 	pt.pause(0.02)
			# 	pt.draw()
			dcdw = self.Constraints.Derivative()
			grad = dcdw.T@ diff
			self.Optim.Corrections(l)
			self.Constraints.Update(grad,self.Optim)
			
			#compute new score
			
			mse = self._ComputeScore(predictPoints)
			
			if mse < minScore:
				minScore = mse
				print(minScore)
				self.Constraints.SavePosition()

				diffnorm = np.linalg.norm(diff)
				dcdwnorm =np.sum(np.abs(dcdw)**2,axis=-1)**(1./2)
				dcdwnorm= dcdwnorm.reshape((len(dcdwnorm),1))
				# angles = (grad / (diffnorm*dcdwnorm))
				# angles[abs(diff) < 1e-2] += np.inf
				# angles[abs(dcdwnorm) < 1e-2] += np.inf
				# angles = np.arccos(angles)
				# angles[~np.isfinite(angles)] = 0
				# diffsave = diff
				# bc = self.Constraints.Vector()
				# print(l,"new best",mse,self.Optim.CurrentAlpha)
				# pt.plot(predictPoints,bc)
				# pt.pause(0.02)
				# pt.draw()
			# else:
			# 	timeSinceDecrease = timeSinceDecrease +1
			# 	# print(l,mse,timeSinceDecrease,minScore)
			# 	if timeSinceDecrease >timeTrigger:
			# 	# 	# print(self.scoreMem,self.deltaScore,self.gradMem)
			# 		self.Constraints._OptimiseVector[:] = minC
			# 	# # 	self.prevScore = minScore
			# 		self.ms = self.ms*0
			# 		self.vs = self.vs *0
			# 		timeTrigger *= 2
			# 		self.MinAlpha /= 2
			# 		currentAlpha = min(0.1*currentAlpha,self.MinAlpha)
			# 		print("TRIGGER")


			l+=1
			self.Optim._CheckConvergence(l,mse,grad)
		
		#set the parameter values to those which achieved the lowest score (in case of large oscillations)
		self.Constraints.RecoverPosition()
		c = self.Constraints.Vector()
		# self.Convexity_Angles = angles
		# self.Convexity_DCDW = dcdwnorm
		# self.Convexity_dLdc = diffsave
	def _ComputeScore(self,predictPoints):
		c = self.Constraints.Vector()
		corrector = self.PseudoInv@(c - self.Constraints.Matrix()@self.p_blups)	
		mse = 0			
		for i in range(len(predictPoints)):
			ai = self.a_blups[i] + corrector[i] * self.epsilon
			mse +=  ai.T @ self.K @ ai - 2 * self.ks[i].T @ai + self.Kernel(predictPoints[i],predictPoints[i])

		if self.Regulariser is not None:
			q = self.Regulariser.F(self.p_clups)
			# print("before",mse,"contrib=",q)
			mse += q
		# print("final",mse)
		return  mse[0,0].astype(float)