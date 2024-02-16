import numpy as np
from pyclup.constraint.vector import *

#each subconstraint contains one matrix and one vector

class SubConstraint:
	IsSub = True
	Matrix = None
	Initialiser = lambda self,ts: None,None
	
	Vector = None
	# Dimension = 0

	def __init__(self,**kwargs):
		self.Matrix = np.zeros((0,0))
		self.Vector = ConstantVector([])
		self.validator= lambda vals: True
		self.validateMessage = ""

		for key,value in kwargs.items():
			if key == "matrix":
				self.Matrix = value
			elif key == "vector":
				self.Vector = value

			
			elif key == "validator":
				self.validator = value
			elif key == "vmessage":
				self.validateMessage = value
			elif key == "initialiser":
				self.Initialiser = value
			else:
				raise KeyError("Unknown key (" + str(key) + ") passed to Constraint Interface")
		self.IsConstant = self.Vector.IsConstant
		self.Dimension = self.Vector.Dimension
		self.TransformDimension = self.Vector.TransformDimension
	
	def InitialiseConstraint(self,ts):
		v,m = self.Initialiser(ts)
		if m is not None:
			self.Vector = v
			self.Matrix = m
		self.IsConstant = self.Vector.IsConstant
		self.Dimension = self.Vector.Dimension
		self.TransformDimension = self.Vector.TransformDimension


	def Transform(self,zs):
		if self.Vector.IsConstant:
			raise RuntimeError("Transform called on a constant constraint - something has gone wrong")
		
		# print("into",self.Vector.Transform(zs))
		return self.Vector.Transform(zs)
	
	def Derivative(self,zs):
		if self.Vector.IsConstant:
			raise RuntimeError("Derivative called on a constant constraint - something has gone wrong")
		# print(zs)
		dcdz = self.Vector.Derivative(zs)
		if np.shape(dcdz)==np.shape(zs):
			dcdz = np.diag(np.reshape(dcdz,len(dcdz),))  #detects if the transform is using a simple separable derivative, converts to diag
		return dcdz

	def Inverse(self,phi):
		return self.Vector.Inverse(phi)

	def Validate(self,predictT):
		passing = self.validator(predictT)
		if not passing:
			raise ValueError("The input data was not as expected for one of the constraints. {self.validateMessage}")