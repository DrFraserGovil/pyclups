import pyclup 
import numpy as np
class Kernel:
	def __init__(self,**kwargs):  
		self.param = [1,1]
		self.function = lambda x, y,param : param[0] * np.exp(-(x - y)**2/(2*param[1]**2))  
		self.variance = 0
		for key,value in kwargs.items():
			if key == "param":
				self.param = value
			elif key == "function":
				self.function = value
			elif key == "variance":
				self.variance = value
			else:
				raise KeyError("Unknown key (" + str(key) + ") passed to kernel")
	def __call__(self,x,y):
		return self.function(x,y,self.param)
	
	def Vector(self,dataT,t):
		out = np.zeros((len(dataT),1))
		
		for i in range(len(dataT)):
			out[i] = self(dataT[i],t)

		return out
	
	def Matrix(self,dataT):
		out = np.zeros((len(dataT),len(dataT)))
		for i in range(len(dataT)):
			for j in range(i,len(dataT)):
				out[i,j] = self(dataT[i],dataT[j])
				
				if i == j:
					out[i,j] += self.variance
				else:
					out[j,i] = out[i,j]
		return out
	
def SquaredExponential(**kwargs):
	sigma = 1
	l0 = 1
	var= 1e-10
	sqex = lambda x, y,param : param[0] * np.exp(-(x - y)**2/(2*param[1]**2))  #the Kernel does currently construct this by default, but can't hurt to futureproof  it/make it explicit
	for key,value in kwargs.items():
		if key == "kernel_variance":
			sigma = value
		elif key == "kernel_scale":
			l0 = value
		elif key == "data_variance":
			var = value
		else:
			raise KeyError("Unknown key (" + str(key) + ") passed to kernel")
	return Kernel(function=sqex,param=[sigma,l0],variance=var)

def Exponential(**kwargs):
	sigma = 1
	l0 = 1
	var= 1e-5
	sqex = lambda x, y,param : param[0] * np.exp(-np.abs(x - y)/(param[1])) 
	for key,value in kwargs.items():
		if key == "kernel_variance":
			sigma = value
		elif key == "kernel_scale":
			l0 = value
		elif key == "data_variance":
			var = value
		else:
			raise KeyError("Unknown key (" + str(key) + ") passed to kernel")
	return Kernel(function=sqex,param=[sigma,l0],variance=var)