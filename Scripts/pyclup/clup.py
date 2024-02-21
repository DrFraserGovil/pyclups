import pyclup
import numpy as np
import emcee
import random
class CLUP:
	trueFunc = None
	
	def __init__(self,kernel,constraint,basis):
		self.Kernel = kernel
		self.Constraints = constraint
		self.Basis = basis

	def Predict(self,predictPoints,dataT,dataX,errorX=1e-20,getErrors=False):
		self.Constraints.Validate(predictPoints)

		self._InitialiseComponents(predictPoints,dataT,dataX,errorX)
		if self.Constraints.IsConstant:
			if np.shape(self.PseudoInv)!=(0,0):

				corrector = self.PseudoInv@(self.Constraints.Vector() - self.Constraints.Matrix()@self.p_blups)				
				for i in range(len(predictPoints)):
					ai = self.a_blups[i] + corrector[i]/self.beta * self.delta
					self.p_clups[i] = ai.T@dataX
			else:
				self.p_clups = self.p_blups

		else:
			self._Optimise(predictPoints)
			corrector = self.PseudoInv@(self.Constraints.Vector() - self.Constraints.Matrix()@self.p_blups)				
			for i in range(len(predictPoints)):
				ai = self.a_blups[i] + corrector[i]/self.beta * self.delta
				self.p_clups[i] = ai.T@dataX
		print(self.p_clups.T)
		e = None
		if getErrors:
			e=self._Errors()
		pred = pyclup.Prediction(predictPoints,self.p_clups,0,self.p_blups,self.p_blps,e)
		pred.blpE = self.blups_error.reshape((len(self.blups_error,)))
		return pred

	def _InitialiseComponents(self,predictPoints,dataT,dataX,errorX):
		eX = np.array(errorX)
		if len(np.atleast_1d(eX)) == 1:
			dataVariance = np.ones(len(dataX)) * errorX**2
		else:
			if len(eX) != len(dataX):
				raise ValueError(f"Data length {len(dataX)} and error length {len(eX)} are not concordant")
			dataVariance= eX**2

	
		dx = np.array(dataX).reshape(-1,1)
		self.Phi = np.zeros((self.Basis.maxOrder+1,len(dataT)))
		for i in range(len(dataT)):
			for m in range(self.Basis.maxOrder+1):
				self.Phi[m,i] = self.Basis(m,dataT[i])

		##scope for some fancy cholesky stuff here -- will do boring way first to get it working
		self.K = self.Kernel.Matrix(dataT,dataVariance)
		self.Kinv = np.linalg.inv(self.K)
		Minv = np.linalg.inv(self.Phi@self.Kinv@self.Phi.T)

		C = self.Kinv@self.Phi.T@Minv
		Delta = (np.eye(len(dataT)) - C@self.Phi).T

		self.ks = [np.zeros((0))]*len(predictPoints)
		self.gammas = [np.zeros((0))]*len(predictPoints)
		self.a_blps = [np.zeros((0))]*len(predictPoints)
		self.a_blups = [np.zeros((0))]*len(predictPoints)
		self.p_blps = np.zeros((len(predictPoints),1))
		self.p_blups = np.zeros((len(predictPoints),1))
		self.p_clups = np.zeros((len(predictPoints),1))
		self.blups_error =np.zeros((len(predictPoints),1))
		self.delta = self.Kinv@Delta@dx
		self.beta = dx.T@self.delta

		self.epsilon = 1.0/self.beta * self.delta

		D = self.Constraints.Matrix()

		self.DDtInv = np.linalg.inv(D@D.T)
		self.PseudoInv = D.T @ self.DDtInv
		
		for i in range(len(predictPoints)):
			self.ks[i] = self.Kernel.Vector(dataT,predictPoints[i])

			self.a_blps[i] = self.Kinv@self.ks[i]
			phi = np.array([self.Basis(j,predictPoints[i]) for j in range(self.Basis.maxOrder+1)]).reshape(-1,1)
			
			ai = Delta.T@self.a_blps[i] + C@phi
			self.a_blups[i] = ai

			self.p_blps[i] = dx.T@self.a_blps[i]
			self.p_blups[i] = dx.T@self.a_blups[i]
			self.blups_error[i] = self.Kernel(predictPoints[i],predictPoints[i]) + ai.T@self.K@ai - 2 * self.ks[i].T@ai


		self._X_data = dx
		self.Dpblub = self.Constraints.Matrix() @ self.p_blups
		self.MSE_Offset = 0
		for i in range(len(dataT)):
			self.MSE_Offset += self.Kernel(dataT[i],dataT[i])
		
	def _Optimise(self, predictPoints):


		#find a good initial position
		
		self.Constraints.InitialPosition(self.p_blups)
		# return
		# return
		# if self.Constraints.c.Revertible:
		# 	cT = self.Dpblub
		# 	self.Constraints.c.Invert(cT)
		ms = np.zeros(shape=np.shape(self.Constraints.TransformDimension,))
		vs = np.zeros(shape=np.shape(self.Constraints.TransformDimension,))
		b1 = 0.7
		b2 = 0.95
		steps = 3000
		alpha = 0.01
		oldScore = 0
		delta = 0
		minScore = self._ComputeScore(predictPoints)
		minC = np.array(self.Constraints._OptimiseVector[:])
		minl = -1
		currentAlpha = alpha
		alphaTrigger = 0
		r = []
		va = []
		for l in range(steps):
			diff = self.DDtInv@(self.Constraints.Vector() - self.Dpblub)

			dcdz = self.Constraints.Derivative()
		
			grad = 2*dcdz@diff

			ms = b1 * ms + (1.0 - b1) * grad
			vs = b2 * vs + (1.0 - b2)*np.multiply(grad,grad)

			c1 = 1.0/(1.0 - pow(b1,l+1))
			c2 = 1.0/(1.0 - pow(b2,l+1))
			step = -currentAlpha*np.divide(ms/c1, np.sqrt(vs/c2 + 1e-20))
			self.Constraints.Update(step)

			gNorm = np.linalg.norm(grad/len(ms))


			s = 0
			pp = 0
			for i in range(len(dcdz)):
				v1 = dcdz[i,:]
				n1 = np.linalg.norm(v1)
				n2 = np.linalg.norm(diff)
				if (n1 > 1e-2 and n2 > 1e-3):
					s += np.dot(v1,diff)/(n1*n2)
					pp+=1
			if (s>0):
				r.append(l)
				va.append(s/(pp+1e-5))
			if gNorm < 1e-7:
				mse = self._ComputeScore(predictPoints)
				print("Reached gnorm at ",l,mse,gNorm)
				break
			if l % 1 == 0:
				mse = self._ComputeScore(predictPoints)
				q= abs(mse - oldScore)/(abs(mse)+1e-7)
				dmem = 0.9
				delta = dmem*delta + (1.0- dmem)*q
				
				
				if minScore == None or mse < minScore:
					minScore = mse
					minC = np.array(self.Constraints._OptimiseVector)
					minl = l+1
				# print(f"Step {l}, score {float(mse)}, best at {minl},{minScore}, gnorm is {gNorm}, {currentAlpha}, {alphaTrigger}")
				if (delta < 1e-10):
					print(f"Reached stability, {delta}")
					break
			mse = self._ComputeScore(predictPoints)
			if l > 5:
				if mse > oldScore:
					alphaTrigger += 2
					if alphaTrigger > 50:
						currentAlpha *= 0.9
						alphaTrigger = 0
				else:
					alphaTrigger = max(0,alphaTrigger-1)
					if alphaTrigger == 0:
						currentAlpha = min(alpha,currentAlpha*1.01)
			oldScore = mse
		print("Best step was found at ",minl,minScore)
		self.Angle = [r,va]
		self.Constraints._OptimiseVector[:] = minC
		pi = []
		corrector = self.PseudoInv@(self.Constraints.Vector() - self.Constraints.Matrix()@self.p_blups)	
		
	def _ComputeScore(self,predictPoints):
		corrector = self.PseudoInv@(self.Constraints.Vector() - self.Constraints.Matrix()@self.p_blups)	
		mse = 0			
		for i in range(len(predictPoints)):
			ai = self.a_blups[i] + corrector[i]/self.beta * self.delta
			mse +=  ai.T @ self.K @ ai - 2 * self.ks[i].T @ai
		return mse + self.MSE_Offset
	

	def ConstraintProject(self,vector):
		Bp = self.Constraints.Matrix() @ vector
		testPhi = Bp - self.Constraints._TotalBaseVector

		phi = np.zeros((len(Bp),1))
		start = 0
		penaliser = 0
		for i in range(len(self.Constraints._internalConstraints)):
			dim = self.Constraints._internalConstraints[i].Dimension
			if not self.Constraints._internalConstraints[i].IsConstant:
				for j in range(dim):
					if testPhi[start+j] > 0:
						phi[start+j] = testPhi[start+j]
					else:
						penaliser += abs(testPhi[start+j])
						
					
			start += dim

		bracket = phi - testPhi
		# print(np.shape(vector),np.shape(self.Constraints.Matrix()),np.shape(bracket),np.shape(self.PseudoInv))
		correct = vector + self.PseudoInv @(bracket)
		# print("original",vector.T)
		# print("transformed",correct.T)
		return correct,penaliser*1000

	def _ErrorScore(self,vector):
		sigma = 0.05
		ps,penalty = self.ConstraintProject(vector.reshape((len(vector),1)))
		
		score = -self.MSE_Offset
		for i in range(len(vector)):
			ai = self.a_blups[i] + (ps[i] - self.p_blups[i])/self.beta * self.delta
			score -= ai.T @self.K @ai - 2*self.ks[i].T @ai
		score -= penalty
		# print("PS",ps.T)
		# print("score",score)
		# print("pen",penalty)
		return score
	def _Errors(self):
		vector = self.p_clups
		
		self.MyVec = vector.reshape((len(vector),1))
		
		print(self._ErrorScore(vector))

		ndim = len(vector)
		nwalkers= int(2.1*ndim)

		startPos = np.tile(vector.T,(nwalkers,1))
		noise = 0.2 * (np.random.random((nwalkers,ndim))-0.5)
		startPos += noise
		for i in range(nwalkers):
			v = startPos[i,:].reshape((ndim,1))
			# print(v)
			t,_ = self.ConstraintProject(v)
			startPos[i,:] = t.T
		sampler = emcee.EnsembleSampler(nwalkers,ndim,lambda ps: self._ErrorScore(ps),moves=[
        (emcee.moves.DEMove(), 0.8),
        (emcee.moves.DESnookerMove(), 0.2),
    ],
		)
		state = sampler.run_mcmc(startPos,5000,progress=True)
		# tau = int(np.mean(np.mean(sampler.get_autocorr_time())))
		try:
			tau = int(np.mean(np.mean(sampler.get_autocorr_time())))
		except:
			tau = 100
		flat_samples = sampler.get_chain(discard=10*tau, thin=2*tau, flat=True)
		random.shuffle(flat_samples)
		print(
			"Mean acceptance fraction: {0:.3f}".format(
				np.mean(sampler.acceptance_fraction)
			)
		)
		print("I have",len(flat_samples),"given",tau)

		# flat_samples = flat_samples[:3]
		out = []
		for j in range(len(flat_samples)):
			ps,pen = self.ConstraintProject(flat_samples[j].reshape((ndim,1)))
			out.append(ps)
		return out