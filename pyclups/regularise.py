import numpy as np

class Regulariser:

	def __init__(self,func,derivative,strength):
		self._Function = func
		self._Derivative = derivative
		self._strength = strength
	
	def F(self,p):
		return self._Function(p)*self._strength

	def dF(self,p):
		return self._strength * self._Derivative(p)



def _smoothingFunction(p):
	s = 0.
	for i in range(1,len(p)):
		s += (p[i] - p[i-1])**2
	return s
def _smoothingGradient(p):
	out = np.zeros(p.shape)

	for i in range(0,len(p)):
		v = 0.
		if i > 0:
			v += 2*(p[i] - p[i-1])
		if i < len(p) - 1:
			v += 2*(p[i] - p[i+1])
		out[i] = v
	return out

def _curvatureFunction(p):
	s = 0.
	for i in range(2,len(p)):
		s += (p[i] - 2*p[i-1] + p[i-2])**2
	return s
def _curvatureGradient(p):
	out = np.zeros(p.shape)
	N = len(p)
	for i in range(0,N):
		v = 0
		if i == 0:
			v = p[0] - 2 * p[1] + p[2]
		elif i == 1:
			v = 5*p[1] - 2 * p[0] - 4 * p[2] + p[3]
		elif i == N-2:
			v = p[i-2] - 4 * p[i-1] + 5 *p[i] - 2 * p[i+1]
		elif i == N-1:
			v = p[i-2] - 2 * p[i-1] + p[i]
		else:
			v = p[i-2] - 4 * p[i-1] + 6*p[i] - 4 *p[i+1] + p[i+2]
		out[i] = v
	return out
def Smoothing(strength):
	return Regulariser(_smoothingFunction,_smoothingGradient,strength)

def Curvature(strength):
	return Regulariser(_curvatureFunction,_curvatureGradient,strength)