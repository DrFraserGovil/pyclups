from pyclup.constraint.constraint_class import *
# from pyclup.constraint
# _vector import *

# class CustomConstraint(pyclup.Constraint):
# 	def __init__(self):


class GreaterThan(Constraint):

	def __init__(self,value,domain=None):
		self.GreaterThan = value
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
		if callable(self.GreaterThan):
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs),self.GreaterThan(ts))
		else:
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs),self.GreaterThan)
		
		vector.SetWBounds(-10,10)
		return vector,matrix

class LessThan(Constraint):

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
	
def Positive(domain=None):
	return GreaterThan(0,domain)
def Negative(domain=None):
	return LessThan(0,domain)


# def Bounded(dataT,valueBelow,valueAbove):

# 	n = len(dataT)
# 	
# 	mat = np.eye(n)
# 	con = Constraint(vector=vec,matrix=mat)
# 	return con


class Bounded(Constraint):

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
		gt = self.GreaterThan
		lt = self.LessThan
		if callable(self.GreaterThan):
			gt = gt(ts)
		if callable(self.LessThan):
			lt = lt(ts)

		vector = OptimiseVector(n,n,lambda zs : (lt-gt)/(1.0 + np.exp(-zs)), lambda zs: (lt-gt)*np.exp(-zs)/(1 + np.exp(-zs))**2, lambda zs: -np.log(np.maximum(1e-8,np.divide(lt-gt,zs)-1)),gt)
		vector.SetWBounds(-10,10)
		return vector,matrix