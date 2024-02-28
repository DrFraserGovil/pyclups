from pyclups.constraint.constraint_class import *

class Bounded(Constraint):
	#it doesn't actually matter if valueBelow and valueAbove are switched around, the code makes sure that it works out in the end
	def __init__(self,valueBelow,valueAbove,domain=None):
		self.GreaterThan = valueBelow
		self.LessThan = valueAbove
		self.Domain = domain
		super().__init__(initialiser=self.InitialiseConstraint)

	def InitialiseConstraint(self,ts):
		n = len(ts)
		ts = ts.reshape(len(ts),1) #hack to make things into the right shape!
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
		gtt = self.GreaterThan
		ltt = self.LessThan
		# lt = np.minimum(lt)
		if callable(self.GreaterThan):
			gtt = gtt(ts)
		if callable(self.LessThan):
			ltt = ltt(ts)
		gt = np.minimum(gtt,ltt)
		lt = np.maximum(gtt,ltt)

		vector = OptimiseVector(n,n,lambda zs : (lt-gt)/(1.0 + np.exp(-zs)), lambda zs: (lt-gt)*np.exp(-zs)/(1 + np.exp(-zs))**2, lambda zs: -np.log(np.maximum(1e-8,np.divide(lt-gt,zs)-1)),gt)
		vector.SetWBounds(-10,10)
		return vector,matrix