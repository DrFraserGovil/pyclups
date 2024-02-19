import pyclup
import numpy as np
import emcee
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
		e = None
		if getErrors:
			e=self._Errors(predictPoints)
		return pyclup.Prediction(predictPoints,self.p_clups,0,self.p_blups,self.p_blps,e)
	

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
			
			
			self.a_blups[i] = Delta.T@self.a_blps[i] + C@phi

			self.p_blps[i] = dx.T@self.a_blps[i]
			self.p_blups[i] = dx.T@self.a_blups[i]
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

	def _ComputeScore(self,predictPoints):
		corrector = self.PseudoInv@(self.Constraints.Vector() - self.Constraints.Matrix()@self.p_blups)	
		mse = 0			
		for i in range(len(predictPoints)):
			ai = self.a_blups[i] + corrector[i]/self.beta * self.delta
			mse +=  ai.T @ self.K @ ai - 2 * self.ks[i].T @ai
		return mse + self.MSE_Offset
	

	def _ErrorScore(self,vector):
		# print("Attempting a score",np.shape(vector))
		vector = vector.reshape((len(vector),1))
		vectordim = len(self._X_data)
		nvecs = int(len(vector)/vectordim)
		ps = np.zeros((nvecs,1))
		start = 0
		for i in range(nvecs):
			a_blup = vector[start:start+vectordim]

			ps[i] = a_blup.T @ self._X_data
			start += vectordim

		Bp = self.Constraints.Matrix()@ps
		cMod = Bp - self.Constraints._TotalBaseVector
		cMod[cMod<0] = 0

		score = self.MSE_Offset
		corrector = self.PseudoInv@(cMod - Bp)
		start = 0
		for i in range(nvecs):
			a_blup = vector[start:start+vectordim]
			ai = a_blup + corrector[i]/self.beta * self.delta
			pi = ai.T@self._X_data
			if (pi<-1e-8):
				print("oh shit",score)
				print(vector.T)
				print("pi",pi)
				r=z
			score += ai.T @ self.K @ ai - 2 * self.ks[i].T @ai
			start += vectordim
		# print(score)
		if np.any(np.abs(vector) > 1e3):
			return -999999999 
		return -score
	def _Errors(self,predictPoints):
		vector = []
		corrector = self.PseudoInv@(self.Constraints.Vector() - self.Constraints.Matrix()@self.p_blups)	
		for i in range(len(predictPoints)):
			ai = self.a_blups[i] + corrector[i]/self.beta * self.delta
			if len(vector)==0:
				vector = ai
			else:	
				vector = np.concatenate((vector,ai))
		
		
		ndim = len(vector)
		nwalkers= 3*ndim

		startPos = np.tile(vector.T,(nwalkers,1))
		noise = 0.00000001 * (np.random.random((nwalkers,ndim))-0.5)
		startPos += noise
		sampler = emcee.EnsembleSampler(nwalkers,ndim,lambda ps: self._ErrorScore(ps))
		state = sampler.run_mcmc(startPos,5000,progress=True)
		try:
			tau = int(np.mean(np.mean(sampler.get_autocorr_time())))
		except:
			tau = 150
		print(tau)
		flat_samples = sampler.get_chain(discard=10*tau, thin=2*tau, flat=True)

		print("I have",len(flat_samples))

		# flat_samples = flat_samples[:3]
		out = []
		for j in range(len(flat_samples)):
			vector = flat_samples[j]
			vector = vector.reshape((len(vector),1))
			print(np.shape(vector))
			vectordim = len(self._X_data)
			nvecs = int(len(vector)/vectordim)
			ps = np.zeros((nvecs,1))
			start = 0
			for i in range(nvecs):
				a_blup = vector[start:start+vectordim]

				ps[i] = a_blup.T @ self._X_data
				start += vectordim
			print(ps.T)
			Bp = self.Constraints.Matrix()@ps
			cMod = Bp - self.Constraints._TotalBaseVector
			cMod[cMod<0] = 0

			corrector = self.PseudoInv@(cMod - Bp)
			start = 0
			for i in range(nvecs):
				a_blup = vector[start:start+vectordim]
				ai = a_blup + corrector[i]/self.beta * self.delta

				pi = ai.T@self._X_data
				print(i,ai,pi)
				ps[i] = pi
				start+=vectordim
			# print(ps)
			out.append(ps)
		return out