import numpy as np


class ConstraintVector:

	def __init__(self,constraintDimension,isConstant):
		self.Dimension = constraintDimension
		self.IsConstant = isConstant
		
		if self.IsConstant:
			self.TransformDimension = 0
		
	def EnforceBounds(self,zs):

		if self.LowerBound is not None:
			zs = np.maximum(zs,self.LowerBound)
		if self.UpperBound is not None:
			zs = np.minimum(zs,self.UpperBound)
		return zs


class ConstantVector(ConstraintVector):
	def __init__(self,values):
		super().__init__(len(values),True)
		self.BaseValue = np.reshape(values,(len(values),1))


class OptimiseVector(ConstraintVector):

	def SetWBounds(self,lower=None,upper=None):

		self.LowerBound = lower
		self.UpperBound = upper
	def __init__(self,dimension,transformDimension,transform,transformDerivative,inverseFunction,offset=None):
		super().__init__(dimension,False)

		self.TransformDimension = transformDimension
		self.Transform = transform
		self.Derivative = transformDerivative
		self.Inverse = inverseFunction
		self.LowerBound = None
		self.UpperBound = None
		if offset is not None:
			if isinstance(offset,int) or isinstance(offset,float):
				self.BaseValue = np.zeros((self.Dimension,1))+offset
			else:
				self.BaseValue = np.reshape(offset,(dimension,1))
		else:
			self.BaseValue = np.zeros((self.Dimension,1))


