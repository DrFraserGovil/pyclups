import pyclup
import numpy as np

class CLUP:
	trueFunc = None
	
	def __init__(self,kernel,constraint,basis):
		self.Kernel = kernel
		self.Constraints = constraint
		self.Basis = basis

	def Predict(self,predictPoints,dataT,dataX,errorX=1e-20):
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
		return pyclup.Prediction(predictPoints,self.p_clups,0,self.p_blups,self.p_blps)
	

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

		self.Dpblub = self.Constraints.Matrix() @ self.p_blups
		# print(np.shape(self.p_blups))

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
		b2 = 0.99
		steps = 3000
		alpha = 0.2
		oldScore = 0
		delta = 0
		minC = np.array(self.Constraints._OptimiseVector[:])
		minScore = self._ComputeScore(predictPoints)
		minl = -1
		r = []
		va = []
		for l in range(steps):
			# print(l,)
			diff = self.DDtInv@(self.Constraints.Vector() - self.Dpblub)

			dcdz = self.Constraints.Derivative()
		
			grad = 2*dcdz@diff
			# print("c",self.Constraints.Vector().T)
			# print("diff",diff.T)
			# print("dcdz",dcdz)
			# print(l,"\n\tpos",self.Constraint.c.Value.T,"\n\tGrad",grad.T)

			ms = b1 * ms + (1.0 - b1) * grad
			vs = b2 * vs + (1.0 - b2)*np.multiply(grad,grad)

			c1 = 1.0/(1.0 - pow(b1,l+1))
			c2 = 1.0/(1.0 - pow(b2,l+1))
			step = -alpha*np.divide(ms/c1, np.sqrt(vs/c2 + 1e-20))
			# print("\tms=",ms.T,"\n\tvs=",vs.T,"\n\tstep=",step.T)
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
				delta = 0.3*delta + (1.0- 0.3)*q
				oldScore = mse
				# print(l,)
				if minScore == None or mse < minScore:
					minScore = mse
					minC = np.array(self.Constraints._OptimiseVector)
					minl = l+1
				# print(l,gNorm,mse,q,delta)
				# print(grad,step,self.Constraints._OptimiseVector)
				if (delta < 1e-10):
					print(f"reached stability, {delta}")
					break
		self.Angle = [r,va]
		print("Average angle",s)
		print("min c achieved at ",minl,minScore)
		self.Constraints._OptimiseVector[:] = minC
			# print("\tnewpos",self.Constraint.c.Value.T)

	def _ComputeScore(self,predictPoints):
		corrector = self.PseudoInv@(self.Constraints.Vector() - self.Constraints.Matrix()@self.p_blups)	
		mse = 0			
		for i in range(len(predictPoints)):
			ai = self.a_blups[i] + corrector[i]/self.beta * self.delta
			
			mse += self.Kernel(predictPoints[i],predictPoints[i]) + ai.T @ self.K @ ai - 2 * self.ks[i].T @ai
		return mse
	
