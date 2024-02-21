import pyclups 
import numpy as np

class Kernel:
	def __init__(self,**kwargs):  
		self.param = [1,1]
		self.function = lambda x, y,param : param[0] * np.exp(-(x - y)**2/(2*param[1]**2))  
		for key,value in kwargs.items():
			if key == "param":
				self.param = value
			elif key == "function":
				self.function = value
			else:
				raise KeyError("Unknown key (" + str(key) + ") passed to kernel")
	def __call__(self,x,y):
		return self.function(x,y,self.param)
	
	def Vector(self,dataT,t):
		out = np.zeros((len(dataT),1))
		
		for i in range(len(dataT)):
			out[i] = self(dataT[i],t)

		return out
	
	def Matrix(self,dataT,data_variance):
		## kernel matrix is equal to K^\prime = K + \sigma^2 I
		out = np.zeros((len(dataT),len(dataT)))
		for i in range(len(dataT)):
			for j in range(i,len(dataT)):
				out[i,j] = self(dataT[i],dataT[j])
				
				if i == j:
					out[i,j] += data_variance[i]
				else:
					out[j,i] = out[i,j]
		return out
	
def SquaredExponential(**kwargs):
	sigma = 1
	l0 = 1

	#the Kernel does currently construct this by default, but can't hurt to futureproof  it/make it explicit
	sqex = lambda x, y,param : param[0] * np.exp(-(x - y)**2/(2*param[1]**2))  
	for key,value in kwargs.items():
		if key == "kernel_variance":
			sigma = value
		elif key == "kernel_scale":
			l0 = value
		else:
			raise KeyError("Unknown key (" + str(key) + ") passed to kernel")
	return Kernel(function=sqex,param=[sigma,l0])

def Exponential(**kwargs):
	sigma = 1
	l0 = 1
	sqex = lambda x, y,param : param[0] * np.exp(-np.abs(x - y)/(param[1])) 
	for key,value in kwargs.items():
		if key == "kernel_variance":
			sigma = value
		elif key == "kernel_scale":
			l0 = value
		else:
			raise KeyError("Unknown key (" + str(key) + ") passed to kernel")
	return Kernel(function=sqex,param=[sigma,l0])