#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as pt
from tqdm import tqdm
import warnings
from scipy import special
import pyclup




np.random.seed(0)
large_width = 400
np.set_printoptions(linewidth=large_width)
warnings.filterwarnings("ignore")

def f(x):
	return np.exp(-x*x/0.1)

bottom = -1
top = 1
[t,x] = pyclup.GenerateData(n=40,mode="semi",xmax=top,xmin=bottom,noise=0.01,function=f)
# tt =np.linspace(min(bottom,min(t)),max(top,max(t)),9)
tt = np.linspace(bottom,top,199)
print(tt)
K = pyclup.kernel.SquaredExponential(kernel_variance=2.5,kernel_scale=0.5)


constraint = pyclup.constraint.Positive(tt, lambda t: t < 0)
constraint2 = pyclup.constraint.Positive(tt, lambda t: t > 0)
constraint.Add(constraint2)

basis = pyclup.basis.Hermite(5)
error_x = 0.1
s = pyclup.clup.CLUP(K,constraint,basis)

pred = s.Predict(tt,t,x,error_x)


pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")
pt.scatter(t,x,label="Sampled Data")
eps = pred.TrueError(f)
pt.plot(pred.T,pred.X_BLUP,label="BLUP $\epsilon=$"+str(pred.blup_error))
pt.plot(pred.T,pred.X,label="CLUP $\epsilon=$"+str(eps))
pt.legend()
pt.draw()
pt.pause(0.01)

input("Enter to exit")