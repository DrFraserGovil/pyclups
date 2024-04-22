from pyclups.constraint.constraint_set import *

class LessThan(ConstraintSet):

	def __init__(self,value,domain=None):
		self.LessThan = value
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
		if callable(self.LessThan):
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs),-1*self.LessThan(ts))
		else:
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs),-1*self.LessThan)
		
		vector.SetWBounds(-10,10)
		return vector,matrix
	
def Negative(domain=None):
	return LessThan(0,domain)