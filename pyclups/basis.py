import numpy as np
import scipy as sp

class Basis:
	def __init__(self):
		self.funcList = []
		self.maxOrder = -1
	def __call__(self, order,x): 
		if order > self.maxOrder:
			raise ValueError("Cannot exceed the basis order")
		return self.funcList[order](x)

def Polynomial(order,mode=None):

	b = Basis()
	b.maxOrder = order
	for i in range(order+1):
		val = i
		if mode == 'odd':
			val = 2*i+1
		if mode == 'eve':
			val = 2 *i
		l = lambda x,n=val: x**n
		b.funcList.append(l)
	return b

def Hermite(order,mode=None):
	b = Basis()
	b.maxOrder = order
	for i in range(order+1):
		val = i
		if mode == 'odd':
			val = 2*i+1
		if mode == 'even':
			val = 2 *i
		l = lambda x,n=val:  sp.special.hermite(n,monic=False)(x)
		b.funcList.append(l)
	return b