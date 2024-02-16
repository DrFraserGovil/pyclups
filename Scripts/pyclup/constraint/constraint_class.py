import numpy as np
from pyclup.constraint.subconstraint import *
##things it should do: generate D, c, transforms
## validate data obeys assumptions of D?


class Constraint:
    
	
	def __init__(self,**kwargs):
		self._internalConstraints = []
		self._TotalMatrix = None
		self._TotalVector = None
		self.TransformDimension = 0
		self.Dimension = 0
		self.IsSub = False
		self.IsConstant = True

		self.Add(SubConstraint(**kwargs))


	def Add(self,constraint):
		if constraint.IsSub:
			self._internalConstraints.append(constraint)
		else:
			self._internalConstraints += constraint._internalConstraints
		
		
	def Matrix(self):
		
		return self._TotalMatrix

	def InitialPosition(self,pblups):
		cInit = self._TotalMatrix@pblups
		phi = cInit - self._TotalBaseVector
		phi[phi<0] = 1e-3

		for i in range(len(self._internalConstraints)):
			if not self._internalConstraints[i].IsConstant:
				lower = self._OptimiserIndices[i][0]
				upper = self._OptimiserIndices[i][1]
				print(phi[lower:upper])
				zsMod = self._internalConstraints[i].Inverse(phi[lower:upper])
				self._OptimiseVector[lower:upper] = zsMod
	def _GenerateMatrix(self):
		self._TotalMatrix = self._internalConstraints[0].Matrix
		self._TotalBaseVector = self._internalConstraints[0].Vector.BaseValue
		for i in range(1,len(self._internalConstraints)):
			self._TotalMatrix = np.concatenate((self._TotalMatrix,self._internalConstraints[i].Matrix),0) 
			self._TotalBaseVector = np.vstack((self._TotalBaseVector,self._internalConstraints[i].Vector.BaseValue))


		self._OptimiseVector = np.zeros((self.TransformDimension,1))
		self._OptimiserIndices = [None]*len(self._internalConstraints)
		start = 0
		for i in range(len(self._internalConstraints)):
			if not self._internalConstraints[i].IsConstant:
				end = start + self._internalConstraints[i].TransformDimension
				# val = [start,end]
				self._OptimiserIndices[i] = [start,end]
				start = end

	def _ComputeVector(self):

		self._TotalVector = np.array(self._TotalBaseVector) #copy by value not reference
		if self.TransformDimension > 0:
			start = 0
			for i in range(len(self._OptimiserIndices)):
				if self._OptimiserIndices[i] != None:
					lower = self._OptimiserIndices[i][0]
					upper = self._OptimiserIndices[i][1]
					dist = upper - lower

					# self._OptimiseVector[start:start+dist] = self._internalConstraints[i].Vector.EnforceBounds(self._OptimiseVector[start:start+dist])
					self._TotalVector[lower:upper] += self._internalConstraints[i].Transform(self._OptimiseVector[start:start+dist])
					start = start + dist
	def Vector(self):
		self._ComputeVector()
		return self._TotalVector
	def Derivative(self):

		self._TotalDerivative = np.zeros((self.TransformDimension,self.Dimension))

		tstart = 0
		dstart = 0

		for i in range(0,len(self._internalConstraints)):
			dim = self._internalConstraints[i].Dimension

			if not self._internalConstraints[i].IsConstant:
				tdim = self._internalConstraints[i].TransformDimension
				self._TotalDerivative[tstart:tstart+tdim,dstart:dstart+dim] += self._internalConstraints[i].Derivative(self._OptimiseVector[tstart:tstart+tdim])
				tstart += tdim

			dstart += dim
		return self._TotalDerivative
	def Update(self,step):
		self._OptimiseVector += step
		
	
	def Validate(self,predictT):

		for i in range(len(self._internalConstraints)):
			self._internalConstraints[i].InitialiseConstraint(predictT)
			self.TransformDimension += self._internalConstraints[i].TransformDimension
			self.Dimension += self._internalConstraints[i].Dimension
		self._GenerateMatrix()
		self.IsConstant = (self.TransformDimension == 0)
	
		shape = np.shape(self._TotalMatrix)
		if shape[0] > 0 and len(predictT) != shape[1]:
			raise ValueError(f"The rows of the constraint matrix ({np.shape(self._TotalMatrix)}) and the number of predictions({len(predictT)}) are not the same")

		for con in self._internalConstraints:
			con.Validate(predictT)
		# dataGood = self.validator(predictT)
		# if not dataGood:
		# 	raise ValueError(f"The provided dataset does not meet the expectations of the Constraint\n\t{self.validateMessage}\n")
		

		if abs(np.linalg.det(self._TotalMatrix@self._TotalMatrix.transpose())) < 1e-8:
			raise ValueError(f"The transpose-product of the constraint matrix has a vanishing determinant. This is likely due to conflicting, simultaneous constraints.")


def Bounded(dataT,valueBelow,valueAbove):

	n = len(dataT)
	vec = OptimiseVector(n,n,lambda zs : valueAbove/(1.0 + np.exp(-zs)), lambda zs: valueAbove*np.exp(-zs)/(1 + np.exp(-zs))**2, lambda zs: -np.log( 1e-8+np.maximum(0,valueAbove/zs-1)),valueBelow)
	mat = np.eye(n)
	con = Constraint(vector=vec,matrix=mat)
	return con

def MonotonicIncreasing(dataT):

	n = len(dataT)
	sort = np.argsort(dataT)
	vec = OptimiseVector(n-1,n-1,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs))

	matrix = np.zeros((n-1,n))

	for i in range(1,n):
		lower = sort[i-1]
		upper = sort[i]
		matrix[i-1][lower] = -1
		matrix[i-1][upper] = 1
	
	con = Constraint(vector=vec,matrix=matrix)
	return con

def Integrable(dataT,value):

	n = len(dataT)

	dx = dataT[1] - dataT[0]

	vec = ConstantVector([value/dx])

	matrix = np.ones((1,n))
	matrix[0,0] = 0.5
	matrix[0,-1] = 0.5
	con = Constraint(vector=vec,matrix=matrix)
	return con