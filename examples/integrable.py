import sys

sys.path.append('../')
import pyclups

import numpy as np
from matplotlib import pyplot as pt
import pyclups
pt.rcParams['text.usetex'] = True
np.random.seed(3)


def gauss(x,mu,var):
	return 1.0/np.sqrt(2*np.pi*var) * np.exp(-0.5*(x-mu)**2/var)
def f(x):
	var = 0.1
	x0 = -1
	x1 = 1.3
	return 0.2*gauss(x,x0,var) + 0.8*gauss(x,x1,var)
	


bottom = -3
top = 3
[t,x] = pyclups.GenerateData(n=20,mode="semi",xmax=top,xmin=bottom,noise=0.05,function=f)
error_x = 0.1

K = pyclups.kernel.SquaredExponential(kernel_variance=1.5,kernel_scale=0.5)
basis = pyclups.basis.Hermite(3)

tt = np.linspace(bottom,top,131)
constraint = pyclups.constraint.Integrable(1)
s = pyclups.Predictor(K,constraint,basis)


pred = s.Predict(tt,t,x,error_x)
eps = pred.TrueError(f)
pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")
pt.scatter(t,x,label="Sampled Data")
pt.plot(pred.T,pred.X_BLUP,label=f"BLUP, $\epsilon={pred.blup_error:.2}$")
pt.plot(pred.T,pred.X,label=f"CLUPS, $\epsilon={eps:.2}$")
pt.legend()
pt.xlabel("$t$")
pt.ylabel("$z$")
pt.title("Integrable Constraint")
pt.savefig("integrable.pdf",format='pdf')

print("blup",np.trapz(pred.X_BLUP,pred.T))
print("blup",np.trapz(pred.X,pred.T))