from pyclups.constraint.constraint_set import *


class Integrable(ConstraintSet):
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
			matrix[0,s[i-1]] += 0.5 
			matrix[0,s[i]] += 0.5
		vector = ConstantVector([self.Integral/dx])
		return vector,matrix

class PositiveIntegrable(ConstraintSet):
	def __init__(self,integral):
		self.Integral = integral
		super().__init__(initialiser=self.InitialiseConstraint)
	def transform(self,ws):
		terms = np.log(self.Abscissa.reshape(len(self.Abscissa),1)) + ws
		sum = terms[0]
		for i in range(1,len(ws)):
			sum = np.maximum(sum,terms[i]) + np.log(1.0 + np.exp(-np.abs(terms[i] - sum)))
		exper = np.exp(ws - sum)
		exper = exper.reshape((len(ws),1))
		return exper
	def deriv(self,ws):
		c = self.Vector()

		dcdw = np.zeros((self.TransformDimension,self.Dimension))
		cdt = np.multiply(self.Abscissa,c).reshape((len(c),))
		for i in range(len(ws)):
			dcdw[i,i] += c[i]
			dcdw[i,:] -= c[i] * cdt
		return dcdw
	def invert(self,cs):
		## technically this is not uniquely invertible, but for the sake of argument, I normalise it such that c[0] = 1, and then go from there. The degeneracy doesn't matter to the optimiser -- this is only used for finding the initial position
		maxVal = np.max(cs)
		ccs = cs/maxVal
		exper = np.log(np.maximum(ccs,1e-3))
		# start = np.maximum(cs[0],1e-3)
		# exper = np.maximum(cs/start,1e-3)
		# exper[exper<1e-8] = 1e-8
		return exper
	def InitialiseConstraint(self,ts):
		n = len(ts)
		ts = ts.reshape(len(ts),1) #hack to make things into the right shape!
		
		matrix = np.eye(n)/self.Integral
		s = np.argsort(ts,0)
		self.Abscissa = np.zeros((n,1))
		for i in range(1,len(ts)):
			lower = ts[s[i-1]]
			upper = ts[s[i]]
			dx = upper - lower			
			self.Abscissa[i,0] += 0.5*dx
			self.Abscissa[i-1,0] += 0.5*dx
		vector = OptimiseVector(n,n,self.transform,self.deriv,self.invert)
		# vector.SetWBounds(-10,10)
		return vector,matrix