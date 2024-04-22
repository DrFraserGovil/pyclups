from pyclups.constraint.constraint_set import *

class Even(ConstraintSet):
	def __init__(self,pivot=0):
		self.Pivot = pivot
		super().__init__(initialiser=self.InitialiseConstraint)

	def InitialiseConstraint(self,ts):
		n = len(ts)
		ts = ts.reshape(len(ts),1) #hack to make things into the right shape!
		
		#Floating point arithmetic might make us miss t_i = -t_j, so do some "closer than the smallest gap between values" trickery. 
		# This might (technically) make a function non-even, since if the domain looks like [-10,-5,0,5.01,10], it would make it so that f(-5) = f(5.01)
		# However, it is close enough that if you were using an even constraint on a domain like that, it really is your fault at that point!
		a = np.sort(ts)
		diffs = np.diff(a,1,0)
		doubleCapture = np.min(diffs)/10 
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