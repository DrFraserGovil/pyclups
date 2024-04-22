import pyclups
import numpy as np
import random
from matplotlib import pyplot as pt
import scipy

class Predictor:
	trueFunc = None
	
	def __init__(self,kernel,constraint,basis,verbose=False):
		self.Kernel = kernel
		self.Constraints = constraint
		self.Basis = basis
		self.Verbose = verbose
		self.alpha = 0.1
		self.optimiser_b1 = 0.7
		self.optimiser_b2 = 0.9
		self.MaxSteps = 500
		self.GradientCriteria = 1e-5
		self.ScoreCriteria =1e-6
		self.AlphaFactor = 10

	def Predict(self,predictPoints,data):
		self.Constraints.Validate(predictPoints)

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
		return pyclups.Prediction(predictPoints,self.p_clups,0,self.p_blups,self.p_blps)

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


	def _Step(self,l,alpha,grad):
		#compute adam step rules
		self.ms = self.optimiser_b1 * self.ms + (1.0 - self.optimiser_b1) * grad
		self.vs = self.optimiser_b2 * self.vs + (1.0 - self.optimiser_b2)*np.multiply(grad,grad)
		c1 = 1.0/(1.0 - pow(self.optimiser_b1,l+1))
		c2 = 1.0/(1.0 - pow(self.optimiser_b2,l+1))
		step = -alpha*np.divide(self.ms/c1, np.sqrt(self.vs/c2 + 1e-20))
		self.Constraints.Update(step)

	def _CheckConvergence(self,l,mse,grad,currentAlpha):
		converged=False
		convMem = 0.9
		earlyCorrector = 1.0/(1.0 - pow(convMem,l+1))
		
		gNorm = np.linalg.norm(grad/len(self.ms))
		self.gradMem = convMem * self.gradMem + (1.0 - convMem) * gNorm

		if self.gradMem * earlyCorrector < self.GradientCriteria:
			if self.Verbose:
				print(f"Convergence Criteria met: Gradient flat ({self.gradMem:.4})")
			converged = True
		alphaCorrector = currentAlpha/self.alpha
		self.scoreMem = convMem * self.scoreMem + (1.0 - convMem) * mse
		scoreDelta = abs((mse - self.scoreMem)/self.scoreMem)
		self.deltaScore = convMem * self.deltaScore + (1.0 - convMem) * scoreDelta
		if self.deltaScore * earlyCorrector < self.ScoreCriteria:
			if self.Verbose:
				print(f"Convergence Criteria met: Function value stable at, {mse:.6} with mean change {100*self.deltaScore:.4}\%",l,alphaCorrector)
			converged = True

		if (l >= self.MaxSteps):
			if self.Verbose:
				print(f"Convergence Criteria met: Took {l} steps")
			converged = True

		## some extensions to the ADAM optimiser to make it take smaller steps when large oscillations
		if l > 5:
			if mse > self.prevScore:
				self.AlphaTrigger += 2
				if self.AlphaTrigger > 10:
					currentAlpha *= 0.1
					self.AlphaTrigger = 0
					if currentAlpha < self.MinAlpha:
						currentAlpha = self.MinAlpha
						self.MinAlpha = self.MinAlpha * 0.99
			else:
				min = -2
				self.AlphaTrigger = self.AlphaTrigger-1
				if self.AlphaTrigger < min:
					self.AlphaTrigger = min
					currentAlpha = currentAlpha * 1.5
					if currentAlpha > self.MaxAlpha:
						currentAlpha = self.MaxAlpha
						self.MaxAlpha = self.MaxAlpha *1.01
		return converged,currentAlpha
	def _Optimise(self, predictPoints):


		#find a good initial position
		self.Constraints.InitialPosition(self.p_blups)
		
		#initialise optimiser parameters
		self.ms = np.zeros(shape=np.shape(self.Constraints.TransformDimension,))
		self.vs = np.zeros(shape=np.shape(self.Constraints.TransformDimension,))

		#values for keeping track of convergence
		self.gradMem = 0
		self.scoreMem = 0
		self.prevScore= 0
		self.deltaScore = 0
		minScore = self._ComputeScore(predictPoints)
		mse = minScore
		minC = np.array(self.Constraints._OptimiseVector[:])
		
		#begin the optimisation loop
		converged = False
		currentAlpha = self.alpha
		self.MinAlpha = currentAlpha/self.AlphaFactor
		self.MaxAlpha = currentAlpha*self.AlphaFactor
		self.AlphaTrigger = 0
		l = 0
		while not converged:
			#compute gradient
			# diff = self.DDtInv@(self.Constraints.Vector() - self.Bp_blub)
			# dcdz = self.Constraints.Derivative()
			grad = 2*self.Constraints.Derivative()@ (self.DDtInv@(self.Constraints.Vector() - self.Bp_blub))
			self._Step(l,currentAlpha,grad)

			#compute new score
			self.prevScore = mse
			mse = self._ComputeScore(predictPoints)
			if mse < minScore:
				minScore = mse
				minC = np.array(self.Constraints._OptimiseVector)

			l+=1
			converged,currentAlpha = self._CheckConvergence(l,mse,grad,currentAlpha)
			# print(l,currentAlpha)
		
		#set the parameter values to those which achieved the lowest score (in case of large oscillations)
		self.Constraints._OptimiseVector[:] = minC
		
	def _ComputeScore(self,predictPoints):
		corrector = self.PseudoInv@(self.Constraints.Vector() - self.Constraints.Matrix()@self.p_blups)	
		mse = 0			
		for i in range(len(predictPoints)):
			ai = self.a_blups[i] + corrector[i] * self.epsilon
			mse +=  ai.T @ self.K @ ai - 2 * self.ks[i].T @ai
		s= mse[0,0].astype(float)
		# print(s)
		return s