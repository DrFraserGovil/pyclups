from pyclup.constraint.constraint_class import *

class GreaterThan(Constraint):

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
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs),self.GreaterThan(ts))
		else:
			vector = OptimiseVector(n,n,lambda zs : np.exp(zs), lambda zs: np.exp(zs), lambda zs: np.log(zs),self.GreaterThan)
		
		vector.SetWBounds(-10,10) #exponentials can get a bit tricky if you are not careful -- the derivative is equal to the exponential of w, so if w is too large or too small, the optimiser can go wrong (either not moving at all, or moving so quickly as to kill the momentum vectors)
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


class Integrable(Constraint):
	def __init__(self,integral):
		self.Integral = integral
		super().__init__(initialiser=self.InitialiseConstraint)

	def InitialiseConstraint(self,ts):
		n = len(ts)
		ts = ts.reshape(len(ts),1) #hack to make things into the right shape!
		
		
		#have to do this a weird way in case the ts are not sorted and 
		#also no guarantee of equal domains, so we'll do it the dumb way
		s = np.argsort(ts,0).reshape((len(ts,)))
		matrix = np.zeros((1,n))
		for i in range(1,len(s)):
			lower = ts[s[i-1]]
			upper = ts[s[i]]
			dx = upper - lower
			matrix[0,s[i-1]] += 0.5 * dx
			matrix[0,s[i]] += 0.5 * dx
		vector = ConstantVector([self.Integral])
		return vector,matrix

class PositiveIntegrable(Constraint):
	def __init__(self,integral):
		self.Integral = integral
		super().__init__(initialiser=self.InitialiseConstraint)
	def transform(self,ws):
		sum = 0
		exper = np.exp(ws)
		for i in range(0,len(exper)):
			
			sum += self.Abscissa[i]*exper[i]
		exper/=sum
		exper = exper.reshape((len(ws),1))
		return exper
	def deriv(self,ws):
		c = self.Vector()
		c = c.reshape((len(c),))
		a = c[0]

		dcdw = np.zeros((self.TransformDimension,self.Dimension))
		for j in range(len(ws)):
			dSdw = -1 * a* c[j] * self.Abscissa[j]
			dcdw[j,:] +=dSdw
			dcdw[j,j] += c[j]
		return dcdw
	def invert(self,cs):
		## technically this is not uniquely invertible, but for the sake of argument, I normalise it such that c[0] = 1, and then go from there. The degeneracy doesn't matter to the optimiser -- this is only used for finding the initial position
		exper = cs/cs[0]
		exper[exper<1e-8] = 1e-8
		return np.log(exper)
	def InitialiseConstraint(self,ts):
		n = len(ts)
		ts = ts.reshape(len(ts),1) #hack to make things into the right shape!
		
		matrix = np.eye(n)/self.Integral
		s = np.argsort(ts,0)
		self.Abscissa = np.zeros((n,))
		for i in range(1,len(ts)):
			lower = ts[s[i-1]]
			upper = ts[s[i]]
			dx = upper - lower			
			self.Abscissa[i] += 0.5*dx
			self.Abscissa[i-1] += 0.5*dx
		vector = OptimiseVector(n,n,self.transform,self.deriv,self.invert)
		vector.SetWBounds(-10,10)
		return vector,matrix

class Even(Constraint):
	def __init__(self,pivot=0):
		self.Pivot = pivot
		super().__init__(initialiser=self.InitialiseConstraint)

	def InitialiseConstraint(self,ts):
		n = len(ts)
		ts = ts.reshape(len(ts),1) #hack to make things into the right shape!
		
		#Floating point arithmetic might make us miss t_i = -t_j, so do some "closer than the smallest gap between values" trickery. 
		# This might (technically) make a function non-even, since if the domain looks like [-10,-5,0,5.01,10], it would make it so that f(-5) = f(5.01)
		# However, it is close enough that if you were using an even constraint on a domain like that, it really is your fault at that point!

		doubleCapture = np.min(np.diff(np.sort(ts+1)))/10 
		diffs = ts - self.Pivot
		pairs = []
		for i in range(len(diffs)):
			if diffs[i] > 0:
				mask = np.abs(diffs+diffs[i]) < doubleCapture
				s =  np.where(mask.any(), mask.argmax(), -1)
				if s > -1:
					pairs.append([i,int(s)])
		
		#have to do this a weird way in case the ts are not sorted and 
		#also no guarantee of equal domains, so we'll do it the dumb way
		matrix = np.zeros((len(pairs),n))
		for i in range(0,len(pairs)):
			lower = pairs[i][0]
			upper = pairs[i][1]

			matrix[i,lower] = 1
			matrix[i,upper] = -1
		vector = ConstantVector([0]*len(pairs))
		return vector,matrix
