import numpy as np
from pyclups.constraint.subconstraint import *

class Constraint:
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
		self.Add(SubConstraint(**kwargs))


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
		phi = cInit - self._TotalBaseVector
		phi[phi<0] = 1e-7 #theoretically this should be zero, but that would rely on us assuming users would always remember to suitably buffer their transforms and not, i.e. just use np.log as the inverse of np.exp
		start = 0
		for i in range(len(self.Constraints)):
			dim = self.Constraints[i].Dimension
			if not self.Constraints[i].IsConstant:
				lower = self._OptimiserIndices[i][0]
				upper = self._OptimiserIndices[i][1]
				zsMod = self.Constraints[i].Inverse(phi[start:start+dim])
				self._OptimiseVector[lower:upper] = zsMod
			start += dim

	def _ConstructElements(self):
		#vertically concatenates the elements of the total matrix and the total base vector
		#In terms of the terminology in the paper:
			#_TotalMatrix is (obviously) the vector B
			#_TotalBaseVector is \vec{\xi} - it contains the offsets for the inexact constraints (such that \psi(w) > 0) and the exact constraints
			#_OptimiseVectir is \vec{w} - the elements of the transform space, such that \vec{c} = \vec{\xi} + \psi(\vec{w})
			#This is all complicated by the fact that the dimensions of OptimiseVector are not equal to the _TotalBaseVector, so we need to keep track of the appropriate slicing indices (which is what _OptimiserIndices does)
		self._TotalMatrix = self.Constraints[0].Matrix
		self._TotalBaseVector = self.Constraints[0].Vector.BaseValue
		for i in range(1,len(self.Constraints)):
			self._TotalMatrix = np.concatenate((self._TotalMatrix,self.Constraints[i].Matrix),0) 
			self._TotalBaseVector = np.vstack((self._TotalBaseVector,self.Constraints[i].Vector.BaseValue))

		
		self._OptimiseVector = np.zeros((self.TransformDimension,1))
		self._OptimiserIndices = [None]*len(self.Constraints)
		start = 0
		for i in range(len(self.Constraints)):
			if not self.Constraints[i].IsConstant:
				end = start + self.Constraints[i].TransformDimension
				self._OptimiserIndices[i] = [start,end]
				start = end

	def _ComputeVector(self):
		#In terms of the paper, this computes #\vec{c}= \vec{\xi} + \psi(\vec{w})
		#Again, the majority of this function is handling the varying dimensionalities of \vec{c} and \vec{w} (and the ordering thereof)
		
		self._TotalVector = np.array(self._TotalBaseVector) #copy by value not reference
		if self.TransformDimension > 0:
			start = 0
			for i in range(len(self._OptimiserIndices)):
				dim = self.Constraints[i].Dimension
				if self._OptimiserIndices[i] != None:
					lower = self._OptimiserIndices[i][0]
					upper = self._OptimiserIndices[i][1]
					dist = upper - lower
					
					self._OptimiseVector[lower:upper] = self.Constraints[i].Vector.EnforceBounds(self._OptimiseVector[lower:upper])

					
					self._TotalVector[start:start+dim] += self.Constraints[i].Transform(self._OptimiseVector[lower:upper])
				start = start + dim
	
	def Vector(self):
		self._ComputeVector()
		return self._TotalVector
	
	def Derivative(self):
		#computes the matrix derivative dc/dw. We make the simplifying constraint that (by construction) subconstraints must be linearly independent, and so \vec{w} can be fully separated by subconstraint. 

		self._TotalDerivative = np.zeros((self.TransformDimension,self.Dimension))
		tstart = 0
		dstart = 0
		for i in range(0,len(self.Constraints)):
			dim = self.Constraints[i].Dimension
			if not self.Constraints[i].IsConstant:
				tdim = self.Constraints[i].TransformDimension
				self._TotalDerivative[tstart:tstart+tdim,dstart:dstart+dim] += self.Constraints[i].Derivative(self._OptimiseVector[tstart:tstart+tdim])
				tstart += tdim
			dstart += dim
		return self._TotalDerivative
	
	def Update(self,step):
		self._OptimiseVector += step	
	
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

		if abs(np.linalg.det(self._TotalMatrix@self._TotalMatrix.transpose())) < 1e-8:
			raise ValueError(f"The transpose-product of the constraint matrix has a vanishing determinant. This is likely due to conflicting, simultaneous constraints.")

