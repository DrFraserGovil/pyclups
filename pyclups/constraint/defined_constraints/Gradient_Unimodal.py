from pyclups.constraint.constraint_set import *

def smoothQ(t,T,delta):
	d = delta/2
	return -1.0 + 2.0/(1 + np.exp((t - T)/delta))
def deltaSmoothQ(t,T,delta):
	d = delta/2
	e =  np.exp((t - T)/d)
	return 2.0/(e**2) * e/delta


class Unimodal(ConstraintSet):
	def __init__(self,domain=None):
		self.Domain = domain
		super().__init__(initialiser=self.InitialiseConstraint)

	
	def transform(self,ws,ts,delta):

		T = ws[-1]
		n = len(ws)-1
		a = np.exp(ws[0:n]) * smoothQ(ts[1:],T,delta)
		corr = -7 + np.log(delta)
		ws[ws<corr] = corr
		# print("w=",ws.T)
		# print("q=",smoothQ(ts[1:],T,delta).T)
		# print("zs=",a.T)
		# r = p
		return a
	def gradient(self,ws,ts,delta):
		n = len(ws) - 1
		T = ws[n]
		out = np.zeros((len(ws)-1,len(ws)))
		for i in range(n):
			t1 =  np.exp(ws[i]) * smoothQ(ts[i+1],T,delta) 
			out[i,i] = t1 
			out[i,n] = np.exp(ws[i]) * deltaSmoothQ(ts[i+1],T,delta) 
		# print("c=",self.transform(ws,ts,delta).T)
		# print(out)
		return out

	def inverse(self,cs,ts,delta):
		p = np.cumsum(cs)
		peak = np.argmax(p)
		print(ts[peak])

		g = -10+np.zeros((len(cs)+1,1))
		for i in range(peak-1,-1,-1):
			diff = max(1e-2*delta,p[i+1] - p[i])
			p[i] = p[i+1] - diff
			g[i] = np.log(diff)
		for i in range(peak + 1,len(cs),1):
			diff = max(1e-2*delta, p[i-1]- p[i])
			p[i] = p[i-1] - diff
			g[i] = np.log(diff)
			print(diff)
		g[-1] = ts[peak]
		return g
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
		dt = ts[s[1]] - ts[s[0]]
		vector = OptimiseVector(constraintDim,constraintDim+1,lambda ws : self.transform(ws,ts,dt), lambda ws: self.gradient(ws,ts,dt), lambda zs: self.inverse(zs,ts,dt))
		vector.SetWBounds(-3,3)
		return vector,matrix
	