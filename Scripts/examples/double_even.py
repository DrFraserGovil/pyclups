import sys

sys.path.append('../')
import pyclup

import numpy as np
from matplotlib import pyplot as pt
import pyclup
import scipy as sp
pt.rcParams['text.usetex'] = True
np.random.seed(1)

def f(x):
	var = 0.1
	return 1.0/np.sqrt(2*np.pi*var) * np.exp(-0.5*x*x/var)


bottom = -2
top = 2
[t,x] = pyclup.GenerateData(n=15,mode="uniform",xmax=top,xmin=bottom,noise=0.1,function=f)
error_x = 0.1

K = pyclup.kernel.SquaredExponential(kernel_variance=1.5,kernel_scale=0.5)

def EvenHermite(order):
	b = pyclup.basis.Basis()
	b.maxOrder = order
	for i in range(order+1):
		l = lambda x,n=2*i:  sp.special.hermite(n,monic=True)(x)
		b.funcList.append(l)

	return b

single_basis = pyclup.basis.Hermite(4)
double_basis = EvenHermite(4) ##we define a custom basis which only includes the even basis elements

tt = np.linspace(bottom,top,131)
constraint = pyclup.constraint.Even()
s = pyclup.clup.CLUP(K,constraint,single_basis)
s2 = pyclup.clup.CLUP(K,constraint,double_basis)

pred = s.Predict(tt,t,x,error_x)
pred2 = s2.Predict(tt,t,x,error_x)
eps = pred.TrueError(f)
eps2 = pred2.TrueError(f)
pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")
pt.scatter(t,x,label="Sampled Data")
pt.plot(pred.T,pred.X_BLUP,label=f"BLUP, $\epsilon={pred.blup_error:.2}$")
pt.plot(pred.T,pred2.X_BLUP,label=f"BLUP, even basis, $\epsilon={pred2.blup_error:.2}$")
pt.plot(pred.T,pred.X,label=f"CLUP, $\epsilon={eps:.2}$")
pt.plot(pred.T,pred2.X,label=f"CLUP, even basis, $\epsilon={eps2:.2}$")
pt.legend()
pt.xlabel("$t$")
pt.ylabel("$z$")
pt.savefig("double_even.pdf",format='pdf')