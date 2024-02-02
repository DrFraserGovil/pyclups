import numpy as np


class ConstraintVector:

	def __init__(self,constraintDimension,isConstant):
		self.Dimension = constraintDimension
		self.IsConstant = isConstant
		
		if self.IsConstant:
			self.TransformDimension = 0
		
	
		# self.zs = np.zeros((constraintDimension,1))
		# self.Transform= transform
		# self._Derivative = transformDerivative
		# self.Value = transform(self.zs)
		# self.Constant = isConstant
		# self.Revertible = isRevertible
		# self._Inverse = inverter
		# self.LowerBound = None
		# self.UpperBound = None
	# def Update(self,step):
	# 	self.zs += step
	# 	if self.LowerBound != None:
	# 		self.zs = np.maximum(self.zs,self.LowerBound)
	# 	if self.UpperBound != None:
	# 		self.zs = np.minimum(self.zs,self.UpperBound)
	# 	self.Value = self.Transform(self.zs)

	# def Invert(self,target):
	# 	self.zs = self._Inverse(target)
	# 	self.Value = self.Transform(self.zs)
	# def Derivative(self):
	# 	return self._Derivative(self.zs)
	


class ConstantVector(ConstraintVector):
	def __init__(self,values):
		super().__init__(len(values),True)
		self.BaseValue = np.reshape(values,(len(values),1))
	# def Value(self):
	# 	return self.BaseValue
class OptimiseVector(ConstraintVector):
	def __init__(self,dimension,transformDimension,transform,transformDerivative,inverseFunction,offset=None):
		super().__init__(dimension,False)

		self.TransformDimension = transformDimension
		self.Transform = transform
		self.Derivative = transformDerivative
		self.Inverse = inverseFunction
		
		if offset == None:
			self.BaseValue = np.zeros((self.Dimension,1))
		else:
			self.BaseValue = np.reshape(offset,(dimension,1))
	
	# def Value(self,zs):
	# 	return self.BaseValue + self.Transform(zs)

