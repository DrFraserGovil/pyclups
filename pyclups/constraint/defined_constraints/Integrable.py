from pyclups.constraint.constraint_class import *


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
		# vector.SetWBounds(-10,10)
		return vector,matrix