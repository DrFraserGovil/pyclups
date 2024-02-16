from pyclup.constraint_class import *
# from pyclup.constraint
# _vector import *

# class CustomConstraint(pyclup.Constraint):
# 	def __init__(self):


class GreaterThan(Constraint):

	def __init__(self,value,domain=None):
		self.GT = value
		self.Domain = domain
		super().__init__(initialiser=self.InitialiseConstraint)

	
	def InitialiseConstraint(self,ts):
		n = len(ts)
		matrix = np.eye(n)
		if self.Domain is not None:
			if callable(self.Domain):
				domainIdx = self.Domain(ts)
				n = np.count_nonzero(domainIdx)
				matrix = np.zeros((n,len(ts))) ##have to redefine matrix so that things go in the right place
				pIdx = 0
				for i in range(len(domainIdx)):
					if domainIdx[i]:
						matrix[pIdx,i] = 1
						pIdx +=1					
				ts = ts[domainIdx]
			else:
				raise(ValueError,"A domain must be specified as a callable function")
		if callable(self.GT):
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs),self.GT(ts))
		else:
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs),self.GT)
		
		vector.SetWBounds(-10,10)
		return vector,matrix

class LessThan(Constraint):

	def __init__(self,value,domain=None):
		self.GT = value
		self.Domain = domain
		super().__init__(initialiser=self.InitialiseConstraint)

	
	def InitialiseConstraint(self,ts):
		n = len(ts)
		matrix = -1.0 * np.eye(n)
		if self.Domain is not None:
			if callable(self.Domain):
				domainIdx = self.Domain(ts)
				n = np.count_nonzero(domainIdx)
				matrix = np.zeros((n,len(ts))) ##have to redefine matrix so that things go in the right place
				pIdx = 0
				for i in range(len(domainIdx)):
					if domainIdx[i]:
						matrix[pIdx,i] = -1
						pIdx +=1					
				ts = ts[domainIdx]
			else:
				raise(ValueError,"A domain must be specified as a callable function")
		if callable(self.GT):
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs),-1*self.GT(ts))
		else:
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs),-1*self.GT)
		
		vector.SetWBounds(-10,10)
		return vector,matrix