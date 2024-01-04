import numpy as np


class Prediction:
	def __init__(self,t,p,efficiency):
		self.T = t
		self.X = p
		self.Efficiency = efficiency

	def TrueError(self,func):
		trueX = func(self.T)
		v = np.sqrt(np.sum((trueX - self.X)**2)/len(trueX))
		return v

