import numpy as np
from pyclups.constraint.constraint import *

class ConstraintSet:
	##constraint class is the user interface for constructing and applying prebuilt (and user-defined) constraints
	##It is actually a container class for a list of subconstraint classes (stored in self.Constraints)
	## this is so that multiple constraints can be easily concatenated without it breaking the simple transform interface
	##The Constraint class manages these subconstraints, and slices up the various 'total vectors' so that the correct components are delivered to the appropriate places
	def __init__(self,**kwargs):
		self.Constraints = []
		self._TotalMatrix = None
		self._TotalVector = None
		self.TransformDimension = 0
		self.Dimension = 0
		self.IsSub = False
		self.IsConstant = True

		## if arguments are present, we pass them along to the appropriate subconstraint constructor
		self.Add(Constraint(**kwargs))


	def Add(self,constraint):
		if constraint.IsSub:
			self.Constraints.append(constraint)
		else:
			self.Constraints += constraint.Constraints
		
	def Matrix(self):
		return self._TotalMatrix

	def InitialPosition(self,pblups):
		##this performs the naive initialisation projection
		## gives a good initial ansatz that is (very often) trivially equal to the true position
		cInit = self._TotalMatrix@pblups
		start = 0
		for i in range(len(self.Constraints)):
			dim = self.Constraints[i].Dimension
			if not self.Constraints[i].IsConstant:
				self.Constraints[i].Inverse(cInit[start:start+dim])
			start += dim

	def _ConstructElements(self):
		#vertically concatenates the elements of the total matrix and the total base vector
		#In terms of the terminology in the paper:
			#_TotalMatrix is (obviously) the vector B
			#_TotalBaseVector is \vec{\xi} - it contains the offsets for the inexact constraints (such that \psi(w) > 0) and the exact constraints
			#This is all complicated by the fact that the dimensions of OptimiseVector are not equal to the _TotalBaseVector, so we need to keep track of the appropriate slicing indices (which is what _OptimiserIndices does)
		self._TotalMatrix = self.Constraints[0].Matrix
		self._TotalBaseVector = self.Constraints[0].Vector.BaseValue
		for i in range(1,len(self.Constraints)):
			self._TotalMatrix = np.concatenate((self._TotalMatrix,self.Constraints[i].Matrix),0) 
			self._TotalBaseVector = np.vstack((self._TotalBaseVector,self.Constraints[i].Vector.BaseValue))
		
		self._ParameterDerivative = np.zeros((self.Dimension,self.TransformDimension))

	def _ComputeVector(self):
		#In terms of the paper, this computes #\vec{c}= \vec{\xi} + \psi(\vec{w})
		#Again, the majority of this function is handling the varying dimensionalities of \vec{c} and \vec{w} (and the ordering thereof)
		
		self._TotalVector = np.array(self._TotalBaseVector) #copy by value not reference
		if self.TransformDimension > 0:
			c_start = 0	
			for i in range(len(self.Constraints)):
				c_dim = self.Constraints[i].Dimension
				self._TotalVector[c_start:c_start+c_dim] = self.Constraints[i].Transform()
				c_start += c_dim
	def SavePosition(self):
		for i in range(len(self.Constraints)):
			self.Constraints[i].SavePosition()
	def RecoverPosition(self):
		for i in range(len(self.Constraints)):
			self.Constraints[i].RecoverPosition()
	def Vector(self):
		self._ComputeVector()
		return self._TotalVector
	
	def Derivative(self):
		#computes the matrix derivative dc/dw. We make the simplifying constraint that (by construction) subconstraints must be linearly independent, and so \vec{w} can be fully separated by subconstraint. 
		self._ParameterDerivative.fill(0.)
	
		c_start = 0
		w_start = 0
		for i in range(len(self.Constraints)):
			c_dim = self.Constraints[i].Dimension
			w_dim = self.Constraints[i].TransformDimension
			if w_dim > 0:
				a = self.Constraints[i].Derivative()
				self._ParameterDerivative[c_start:c_start+c_dim, w_start:w_start+w_dim] = a
			c_start += c_dim
			w_start += w_dim
		# print(self._ParameterDerivative)
		# r = p
		return self._ParameterDerivative

	def Update(self,grad,Optim):
		start = 0
		for i in range(len(self.Constraints)):
			dim = self.Constraints[i].TransformDimension
			if dim > 0:
				self.Constraints[i].Update(grad[start:start+dim],Optim)
				start += dim

	def Validate(self,predictT):
		self.TransformDimension = 0
		self.Dimension = 0
		for i in range(len(self.Constraints)):
			self.Constraints[i].InitialiseConstraint(predictT)
			self.TransformDimension += self.Constraints[i].TransformDimension
			self.Dimension += self.Constraints[i].Dimension

		self._ConstructElements()
		self.IsConstant = (self.TransformDimension == 0)
	
		shape = np.shape(self._TotalMatrix)
		if shape[0] > 0 and len(predictT) != shape[1]:
			raise ValueError(f"The rows of the constraint matrix ({np.shape(self._TotalMatrix)}) and the number of predictions({len(predictT)}) are not the same")

		#this allows constraints to have their own validation on assumptions
		for con in self.Constraints:
			con.Validate(predictT)

	
		if abs(np.linalg.cond(self._TotalMatrix@self._TotalMatrix.transpose())) < 1e-8:
			raise ValueError(f"The transpose-product of the constraint matrix has a vanishing determinant. This is likely due to conflicting, simultaneous constraints.")

	def Remove(self):
		self.Constraints = self.Constraints[:-1]
	def BulkUp(self,predictT):
		gamma = np.zeros(predictT.shape)
		B = self.Matrix()
		brows,bcols = B.shape
		lastZeroIdx = len(predictT)-1
		remaining = lastZeroIdx + 1
		for i in range(brows):
			j = lastZeroIdx
			while (j>=0):
				if B[i,j] != 0 and gamma[j] == 0:
					gamma[j] = 1
					remaining -=1
					while lastZeroIdx >= 0 and gamma[lastZeroIdx] == 1:
						lastZeroIdx -=1
					break
				j-=1
		# print(B)
		# print(gamma)
		unconstrained = np.nonzero(1-gamma)[0]
		U = np.zeros((remaining,len(predictT)))
		for i in range(remaining):
			
			U[i,unconstrained[i]] = 1
		
		trivial = lambda x: x
		dtrivial = lambda x: x*0+1

		bulk_vector = OptimiseVector(remaining,remaining,trivial,dtrivial,trivial)

		newcon = Constraint(matrix=U,vector=bulk_vector)
		self.Add(newcon)
		self.Validate(predictT)