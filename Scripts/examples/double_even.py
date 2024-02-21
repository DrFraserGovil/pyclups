import sys

sys.path.append('../')

import numpy as np
from matplotlib import pyplot as pt
import pyclups
import scipy as sp
pt.rcParams['text.usetex'] = True
np.random.seed(3)

def f(x):
	var = 0.1
	return 1.0/np.sqrt(2*np.pi*var) * np.exp(-0.5*x*x/var)


bottom = -2
top = 2
[t,x] = pyclups.GenerateData(n=15,mode="semi",xmax=top,xmin=bottom,noise=0.1,function=f)
error_x = 0.1

K = pyclups.kernel.SquaredExponential(kernel_variance=1.5,kernel_scale=0.6)

single_basis = pyclups.basis.Hermite(4)
double_basis = pyclups.basis.Hermite(2,'even')

tt = np.linspace(bottom,top,131)
constraint = pyclups.constraint.Even()
s = pyclups.Predictor(K,constraint,single_basis)
s2 = pyclups.Predictor(K,constraint,double_basis)

pred = s.Predict(tt,t,x,error_x)
pred2 = s2.Predict(tt,t,x,error_x)
eps = pred.TrueError(f)
eps2 = pred2.TrueError(f)
pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")
pt.scatter(t,x,label="Sampled Data")
pt.plot(pred.T,pred.X_BLUP,label=f"BLUP, $\epsilon={pred.blup_error:.2}$")
pt.plot(pred.T,pred2.X_BLUP,label=f"BLUP, even basis, $\epsilon={pred2.blup_error:.2}$")
pt.plot(pred.T,pred.X,label=f"CLUPS, $\epsilon={eps:.2}$")
pt.plot(pred.T,pred2.X,label=f"CLUPS, even basis, $\epsilon={eps2:.2}$")
pt.title("Doubly-Even Comparison")
pt.legend()
pt.xlabel("$t$")
pt.ylabel("$z$")
pt.savefig("double_even.pdf",format='pdf')