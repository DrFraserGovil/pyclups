#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as pt
from tqdm import tqdm
import warnings
from scipy import special
import pyclup




np.random.seed(3)
large_width = 400
np.set_printoptions(linewidth=large_width)
warnings.filterwarnings("ignore")

def f(x):
	return 1.0/np.sqrt(2*np.pi*0.1) * np.exp(-0.5*x*x/0.1)

	# return 1.0/(1 + np.exp(-x*3))


bottom = -2
top = 2
[t,x] = pyclup.GenerateData(n=40,mode="semi",xmax=top,xmin=bottom,noise=0.15,function=f)
error_x = 0.1

K = pyclup.kernel.SquaredExponential(kernel_variance=2.5,kernel_scale=0.5)
basis = pyclup.basis.Hermite(3)

constraint = pyclup.constraint.GreaterThan(lambda ts: 0.25*ts +0.5, lambda ts: ts>0)
constraint2 = pyclup.constraint.LessThan(lambda ts: 0.1*ts +0.1, lambda ts: ts<=0)
constraint.Add(constraint2)

s = pyclup.clup.CLUP(K,constraint,basis)


tt = np.linspace(bottom,top,190)
pred = s.Predict(tt,t,x,error_x)


pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")
pt.scatter(t,x,label="Sampled Data")
eps = pred.TrueError(f)
pt.plot(pred.T,pred.X_BLUP,label="BLUP $\epsilon=$"+str(pred.blup_error))
pt.plot(pred.T,pred.X,label="CLUP $\epsilon=$"+str(eps))

print("Func",np.trapz(f(tt),tt))
print("BLUP",np.trapz(pred.X_BLUP,pred.T))
print("CLUP",np.trapz(pred.X,pred.T))

pt.legend()
pt.draw()
pt.pause(0.01)

input("Enter to exit")