import pyclup
import numpy as np
class Prediction:
	T = None
	BLP = None
	BLUP = None
	CLUP = None
	def __init__(self,t,blp,blup,clup):
		self.T = t
		self.BLP = blp
		self.BLUP = blup
		self.CLUP = clup

class CLUP:
    
	def __init__(self,kernel,constraint,basis):
		self.Kernel = kernel
		self.Constraint = constraint
		self.Basis = basis
		self.BasisOrder = 4

	def Predict(self,predictPoints,dataT,dataX):
		self.Constraint.Validate(predictPoints)


		self.InitialiseComponents(predictPoints,dataT,dataX)

		if self.Constraint.c.Constant:
			if np.shape(self.PseudoInv)!=(0,0):

				corrector = self.PseudoInv@(self.Constraint.c.Value - self.Constraint.D@self.p_blups)				
				for i in range(len(predictPoints)):
					ai = self.a_blups[i] + corrector[i]/self.beta * self.delta
					self.p_clups[i] = ai.T@dataX
			else:
				self.p_clups = self.p_blups
		return Prediction(predictPoints,self.p_blps,self.p_blups,self.p_clups)
	def InitialiseComponents(self,predictPoints,dataT,dataX):
		print("initialising")
		dx = np.array(dataX).reshape(-1,1)
		self.Phi = np.zeros((self.BasisOrder+1,len(dataT)))
		for i in range(len(dataT)):
			for m in range(self.BasisOrder+1):
				self.Phi[m,i] = self.Basis(m,dataT[i])

		##scope for some fancy cholesky stuff here -- will do boring way first to get it working
		self.K = self.Kernel.Matrix(dataT)
		self.Kinv = np.linalg.inv(self.K)
		Minv = np.linalg.inv(self.Phi@self.Kinv@self.Phi.T)

		C = self.Kinv@self.Phi.T@Minv
		Delta = (np.eye(len(dataT)) - C@self.Phi).T

		self.ks = [np.zeros((0))]*len(predictPoints)
		self.a_blps = [np.zeros((0))]*len(predictPoints)
		self.a_blups = [np.zeros((0))]*len(predictPoints)
		self.p_blps = np.zeros((len(predictPoints),1))
		self.p_blups = np.zeros((len(predictPoints),1))
		self.p_clups = np.zeros((len(predictPoints),1))
		self.delta = self.Kinv@Delta@dx
		self.beta = dx.T@self.delta

		self.PseudoInv = self.Constraint.D.T @ np.linalg.inv(self.Constraint.D@self.Constraint.D.T)

		for i in range(len(predictPoints)):
			self.ks[i] = self.Kernel.Vector(dataT,predictPoints[i])

			self.a_blps[i] = self.Kinv@self.ks[i]
			phi = np.array([self.Basis(j,predictPoints[i]) for j in range(self.BasisOrder+1)]).reshape(-1,1)
			

			self.a_blups[i] = Delta.T@self.a_blps[i] + C@phi

			self.p_blps[i] = dx.T@self.a_blps[i]
			self.p_blups[i] = dx.T@self.a_blups[i]
		# print(np.shape(self.p_blups))

			
