from pyclups.constraint.constraint_set import *
import numpy as np
class GreaterThan(ConstraintSet):

	def __init__(self,value,domain=None):
		self.GreaterThan = value
		self.Domain = domain
		super().__init__(initialiser=self.InitialiseConstraint)

	
	def InitialiseConstraint(self,ts):
		n = len(ts)
		matrix = np.eye(n)
		
		#lets us apply the constraint on only parts of ts, according to a boolean function provided at construction
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
		if callable(self.GreaterThan):
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(np.maximum(zs,1e-3)),self.GreaterThan(ts))
		else:
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(np.maximum(zs,1e-3)),self.GreaterThan)
		
		vector.SetWBounds(-10,10) #exponentials can get a bit tricky if you are not careful -- the derivative is equal to the exponential of w, so if w is too large or too small, the optimiser can go wrong (either not moving at all, or moving so quickly as to kill the momentum vectors)
		return vector,matrix
	

def Positive(domain=None):
	return GreaterThan(0,domain)