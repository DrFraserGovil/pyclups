import numpy as np


class Prediction:
	def __init__(self,t,p,efficiency,blup=None,blp=None):
		self.T = t
		self.X = np.reshape(p,(len(p),))
		if type(blup) != type(None):
			self.X_BLUP = np.reshape(blup,(len(blup),))
		if type(blp) != type(None):
			self.X_BLP = np.reshape(blp,(len(blp),))
		self.Efficiency = efficiency
		

	def TrueError(self,func):
		trueX = func(self.T)
		v = np.sqrt(np.sum((trueX - self.X)**2)/len(trueX))
		if type(self.X_BLUP) != type(None):
			self.blup_error = np.sqrt(np.sum((trueX - self.X_BLUP)**2)/len(trueX))
		if type(self.X_BLP) != type(None):
			self.blp_error = np.sqrt(np.sum((trueX - self.X_BLP)**2)/len(trueX))
		return v

