import sys

sys.path.append('../')
import pyclup

import numpy as np
from matplotlib import pyplot as pt
import pyclup
pt.rcParams['text.usetex'] = True
np.random.seed(3)

def f(x):
	var = 0.1
	return 1.0/np.sqrt(2*np.pi*var) * np.exp(-0.5*x*x/var)


bottom = -3
top = 3
[t,x] = pyclup.GenerateData(n=20,mode="semi",xmax=top,xmin=bottom,noise=0.05,function=f)
error_x = 0.1

K = pyclup.kernel.SquaredExponential(kernel_variance=1.5,kernel_scale=0.5)
basis = pyclup.basis.Hermite(3)

tt = np.linspace(bottom,top,131)
constraint = pyclup.constraint.Even()
constraint.Add(pyclup.constraint.Positive(lambda t:t>0)) #have to apply the constraint only to t>0 to ensure that the constraint dimensions are not in excess

s = pyclup.clup.CLUP(K,constraint,basis)

pred = s.Predict(tt,t,x,error_x)
eps = pred.TrueError(f)
pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")
pt.scatter(t,x,label="Sampled Data")
pt.plot(pred.T,pred.X_BLUP,label=f"BLUP, $\epsilon={pred.blup_error:.2}$")
pt.plot(pred.T,pred.X,label=f"CLUP, $\epsilon={eps:.2}$")
pt.legend()
pt.xlabel("$t$")
pt.ylabel("$z$")
pt.savefig("positive_and_even.pdf",format='pdf')