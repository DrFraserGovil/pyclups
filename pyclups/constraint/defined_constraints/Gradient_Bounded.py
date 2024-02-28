from pyclups.constraint.constraint_class import *

class BoundedGradient(Constraint):
	#it doesn't actually matter if valueBelow and valueAbove are switched around, the code makes sure that it works out in the end
	def __init__(self,lowerBound,upperBound,domain=None):
		self.GreaterThan = upperBound
		self.LessThan = lowerBound
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
				matrix[pIdx,myIdx] = -1
				matrix[pIdx,nextIdx] = 1 
				pIdx +=1	
		gtt = self.GreaterThan
		ltt = self.LessThan
		# lt = np.minimum(lt)
		if callable(self.GreaterThan):
			gtt = gtt(ts)
		if callable(self.LessThan):
			ltt = ltt(ts)
		gt = np.minimum(gtt,ltt)
		lt = np.maximum(gtt,ltt)

		#assuming uniform absiscca here
		dt = float(ts[s[1]] - ts[s[0]])

		vector = OptimiseVector(n-1,n-1,lambda zs : dt*(lt-gt)/(1.0 + np.exp(-zs)), lambda zs: dt*(lt-gt)*np.exp(-zs)/(1 + np.exp(-zs))**2, lambda zs: -np.log(np.maximum(1e-8,np.divide(dt*(lt-gt),zs)-1)),gt*dt)
		vector.SetWBounds(-10,10)
		return vector,matrix
	
class PositiveBoundedGradient(Constraint):
	#it doesn't actually matter if valueBelow and valueAbove are switched around, the code makes sure that it works out in the end
	def __init__(self,lowerBound,upperBound,domain=None):
		self.GreaterThan = upperBound
		self.LessThan = lowerBound
		self.Domain = domain
		super().__init__(initialiser=self.InitialiseConstraint)

	def transform(self,ws,lt,gt):

		out = np.zeros(np.shape(ws))
		out[0] = np.exp(ws[0])
		for i in range(1,len(ws)):
			lower = max(gt+out[i-1],0)
			upper = lt + out[i-1]

			out[i] = lower + (upper-lower)/(1.0 + np.exp(-ws[i]))
		return out
	def inverse(self,ps,lt,gt):
		out = np.zeros(np.shape(ps))
		out[0] = np.log(max(1e-10,ps[0]))
		for i in range(1,len(ps)):
			lower = max(gt+ps[i-1],0)
			upper = lt+ps[i-1]

			t = (ps[i] - lower)/(upper - ps[i])
			out[i] = np.log(max(1e-10,t))
		return out
	
	def derivative(self,ws,lt,gt):
		n = len(ws)
		out = np.zeros((n,n))
		exper = np.exp(-ws)
		prev = 0
		for j in range(n):
			prev = self._TotalVector[j]
			if j == 0:
				prev = np.exp(ws[j])
				out[j,j] = prev
			else:
				lower = max(gt+prev,0)
				upper = lt + prev
				div = 1.0/(1.0 + exper[j])
				out[j,j] = (upper - lower) * div * div * exper[j]

			for i in range(j+1,n):
				prev = self._TotalVector[i]
				mult = 1
				if gt + prev < 0:
					mult = 1.0/(1.0 + exper[j])
				out[j,i] = mult * out[j,i-1]
				
		return out

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

		constraintDim = n
		matrix = np.eye(constraintDim)
		gtt = self.GreaterThan
		ltt = self.LessThan
		# lt = np.minimum(lt)
		if callable(self.GreaterThan):
			gtt = gtt(ts)
		if callable(self.LessThan):
			ltt = ltt(ts)
		dt = float(ts[s[1]] - ts[s[0]])
		gt = np.minimum(gtt,ltt)*dt
		lt = np.maximum(gtt,ltt)*dt

		#assuming uniform absiscca here

		vector = OptimiseVector(n,n,lambda zs : self.transform(zs,lt,gt), lambda zs: self.derivative(zs,lt,gt), lambda zs: self.inverse(zs,lt,gt),0)
		vector.SetWBounds(-10,10)
		return vector,matrix