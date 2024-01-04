#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as pt
from tqdm import tqdm
import warnings
from scipy import special
import pyclup


large_width = 400
np.set_printoptions(linewidth=large_width)
warnings.filterwarnings("ignore")

def f(x):
	return np.exp(-x*x/0.1)

bottom = -1
top = 1
[t,x] = pyclup.GenerateData(n=40,mode="semi",xmax=top,xmin=bottom,noise=0.01,function=f)
tt =np.linspace(min(bottom,min(t)),max(top,max(t)),1000)
K = pyclup.kernel.SquaredExponential(kernel_variance=2.5,kernel_scale=0.5)
constraint = pyclup.constraint.Positive(len(tt))
basis = lambda i,t : special.hermite(i,monic=True)(t)
error_x = 0.1
s = pyclup.clup.CLUP(K,constraint,basis)

p = pyclup.Prediction(t,x,0.1)

pred = s.Predict(tt,t,x,error_x)


pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")
pt.scatter(t,x,label="Sampled Data")
pt.plot(pred.T,pred.BLP,label="Prediction, \epsilon="+str(p.TrueError(f)))
pt.plot(pred.T,pred.CLUP,label="Prediction, \epsilon="+str(p.TrueError(f)))
pt.legend()
pt.draw()
pt.pause(0.01)

input("Enter to exit")