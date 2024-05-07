import numpy as np


class ConstraintVector:

	def __init__(self,constraintDimension,isConstant):
		self.Dimension = constraintDimension
		self.IsConstant = isConstant
		
		if self.IsConstant:
			self.TransformDimension = 0
	
	def SavePosition(self):
		if not self.IsConstant:
			self.BestW = self.ws
	def RecoverPosition(self):
		if not self.IsConstant:
			self.ws = self.BestW

class ConstantVector(ConstraintVector):
	def __init__(self,values):
		super().__init__(len(values),True)
		self.BaseValue = np.reshape(values,(len(values),1))


class OptimiseVector(ConstraintVector):

	def SetWBounds(self,lower=None,upper=None):

		self.LowerBound = lower
		self.UpperBound = upper
	def Transform(self):
		return self._Transform(self.ws)
	
	def Inverse(self,cs):
		self.ws = self._Inverse(cs)

		self.ws[~np.isfinite(self.ws)] = 0 #quick and dirty fix
	def Derivative(self):
		return self._Derivative(self.ws)
	
	def Update(self,grad,Optim):
		self.ms = Optim.b1 * self.ms + (1.0 - Optim.b1) * grad
		self.vs = Optim.b2 * self.vs + (1.0 - Optim.b2) * np.multiply(grad,grad)
		step = Optim.CurrentAlpha*np.divide(self.ms/Optim.c1, np.sqrt(self.vs/Optim.c2 + 1e-20))
		# print("step=",step[:10].T,self.ws.shape,Optim.CurrentAlpha)
		self.ws -= step
	def __init__(self,dimension,transformDimension,transform,transformDerivative,inverseFunction,offset=None):
		super().__init__(dimension,False)

		self.TransformDimension = transformDimension
		self._Transform = transform
		self._Derivative = transformDerivative
		self._Inverse = inverseFunction
		self.LowerBound = None
		self.UpperBound = None
		if offset is not None:
			if isinstance(offset,int) or isinstance(offset,float):
				self.BaseValue = np.zeros((self.Dimension,1))+offset
			else:
				self.BaseValue = np.reshape(offset,(dimension,1))
		else:
			self.BaseValue = np.zeros((self.Dimension,1))

		##initialise the optimisation values
		self.ws = np.zeros((self.TransformDimension,1))
		self.ms  = np.zeros((self.TransformDimension,1))
		self.vs  = np.zeros((self.TransformDimension,1))

