import pyclup
import numpy as np
##things it should do: generate D, c, transforms
## validate data obeys assumptions of D?


#private variables -- think about
class ConstraintVector:

	def __init__(self,constraintDimension,transform,transformDerivative,isConstant,isRevertible,inverter=None):
		self.Dim = constraintDimension
		self.zs = np.zeros((constraintDimension,1))
		self.Transform= transform
		self._Derivative = transformDerivative
		self.Value = transform(self.zs)
		self.Constant = isConstant
		self.Revertible = isRevertible
		self._Inverse = inverter
		self.LowerBound = None
		self.UpperBound = None
	def Update(self,step):
		self.zs += step
		if self.LowerBound != None:
			self.zs = np.maximum(self.zs,self.LowerBound)
		if self.UpperBound != None:
			self.zs = np.minimum(self.zs,self.UpperBound)
		self.Value = self.Transform(self.zs)

	def Invert(self,target):
		self.zs = self._Inverse(target)
		self.Value = self.Transform(self.zs)
	def Derivative(self):
		return self._Derivative(self.zs)
class ConstantVector(ConstraintVector):
	def __init__(self,values):
		self.Dim = len(values)
		self.Constant = True
		self.Value = np.reshape(values,(len(values),1))

class OptimiseVector(ConstraintVector):
	def __init__(self,dimension,transform,transformDerivative,revert=False,revFunc=None):
		self.Dim = dimension
		self.zs = np.zeros((dimension,1))
		self.Transform = transform
		self._Derivative = transformDerivative
		self.Constant = False
		self.Revertible=revert
		self.Value = self.Transform(self.zs)
		self._Inverse = revFunc


	


class Constraint:
    
	def __init__(self,**kwargs):
		self.D = np.zeros((0,0))
		self.c = ConstantVector([])
		self.validator= lambda vals: True
		self.validateMessage = ""

		for key,value in kwargs.items():
			if key == "matrix":
				self.D = value
			elif key == "vector":
				self.c = value
			elif key == "validator":
				self.validator = value
			elif key == "vmessage":
				self.validateMessage = value	
			else:
				raise KeyError("Unknown key (" + str(key) + ") passed to kernel")


	def Validate(self,predictT):
		shape = np.shape(self.D)
		if shape[0] > 0 and len(predictT) != shape[1]:
			raise ValueError(f"The rows of the constraint matrix ({np.shape(self.D)}) and the number of predictions({len(predictT)}) are not the same")

		dataGood = self.validator(predictT)
		if not dataGood:
			raise ValueError(f"The provided dataset does not meet the expectations of the Constraint\n\t{self.validateMessage}\n")
		

		if abs(np.linalg.det(self.D@self.D.transpose())) < 1e-8:
			raise ValueError(f"The transpose-product of the constraint matrix has a vanishing determinant. This is likely due to conflicting, simultaneous constraints.")



def Positive(n):
	
	c = OptimiseVector(n,lambda z: np.exp(z), lambda z: np.exp(z),True,lambda f: np.log(np.maximum(f,1e-9)))
	c.LowerBound =-10
	c.UpperBound = 10
	D = np.eye(n)
	return Constraint(vector=c,matrix=D)

def Integrable(ts,integral):
	c = ConstantVector([integral])
	D = np.zeros((1,len(ts)))
	
	#assumes ts sorted, but not equidistant
	for i in range(len(ts)-1):
		dx= 0.5 * (ts[i+1] - ts[i])
		D[0,i] += dx
		D[0,i+1] += dx

	return Constraint(vector=c,matrix=D,validator=lambda a: np.all(a[:-1] <= a[1:]),vmessage="The integrable constraint only works if the prediction points are sorted")
def Monotonic(ts):
	c = OptimiseVector(len(ts)-1,lambda z: np.exp(z), lambda z: np.exp(z)*0+1,True,lambda f: np.log(np.maximum(f,1e-1800))) #this contains a disgusting hack -- that the gradient is killed by the exp(z) term and removing it doesn't change the direction of optimisation, but does speed it up....do we care since the parameters are individually optimised in ADAM?
	c.LowerBound = -20
	c.UpperBound =20
	D = np.zeros((len(ts)-1,len(ts)))
	for i in range(len(ts)-1):
		D[i,i] = -1
		D[i,i+1] = 1
	return Constraint(vector=c,matrix=D,validator=lambda a: np.all(a[:-1] <= a[1:]),vmessage="The monotonic constraint only works if the prediction points are sorted")

