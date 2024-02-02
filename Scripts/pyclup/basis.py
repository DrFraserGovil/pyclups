import numpy as np
import scipy as sp

class Basis:

	funcList = []
	maxOrder = -1

	def __call__(self, order,x): 
		if order > self.maxOrder:
			raise ValueError("Cannot exceed the basis order")
		# print("outputting",order,self.funcList[order])
		return self.funcList[order](x)
	


def Polynomial(order):

	b = Basis()
	b.maxOrder = order
	for i in range(order+1):
		l = lambda x,n=i: x**n
		# print("assigning",l(2))
		b.funcList.append(l)

	return b

def Hermite(order):
	b = Basis()
	b.maxOrder = order
	for i in range(order+1):
		l = lambda x,n=i:  sp.special.hermite(n,monic=False)(x)
		b.funcList.append(l)

	return b