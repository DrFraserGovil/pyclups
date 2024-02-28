from pyclups.constraint.constraint_class import *

class Monotonic(Constraint):
	#direction = 1 is monotonic increasing , direction = -1 is monotonic decreasing 
	def __init__(self,direction=1,domain=None):
		self.Direction = direction
		self.Domain = domain
		super().__init__(initialiser=self.InitialiseConstraint)

	def InitialiseConstraint(self,ts):
		n = len(ts)
		ts = ts.reshape(len(ts),1) #hack to make things into the right shape!
		domainIdx = [True]*len(ts)
		if self.Domain is not None:
			if callable(self.Domain):
				domainIdx = self.Domain(ts)
				n = np.count_nonzero(domainIdx)
							
				# ts = ts[domainIdx]
			else:
				raise(ValueError,"A domain must be specified as a callable function")
		
		#have to do this a weird way in case a) the ts are not sorted and b) the domain is active	 
		s = np.argsort(ts,0).reshape((len(ts,)))

		constraintDim = n-1
		matrix = np.zeros((constraintDim,len(domainIdx)))

		pIdx = 0
		for i in range(0,len(s)-1):
			myIdx = s[i]
			nextIdx = s[i+1]
			if domainIdx[myIdx]:
				matrix[pIdx,myIdx] = -1*self.Direction
				matrix[pIdx,nextIdx] = self.Direction 
				pIdx +=1	
		vector = OptimiseVector(constraintDim,constraintDim,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs),0)
		vector.SetWBounds(-10,10)
		return vector,matrix
	