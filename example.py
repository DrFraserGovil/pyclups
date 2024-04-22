#!/opt/homebrew/bin/python3

import os
os.environ['OPENBLAS_NUM_THREADS'] = "1"
import numpy as np
from tqdm import tqdm
import warnings
from scipy import special
from matplotlib import pyplot as pt
import pyclups

import sys
np.set_printoptions(threshold=sys.maxsize)

np.random.seed(2)
large_width = 400
np.set_printoptions(linewidth=large_width)
warnings.filterwarnings("ignore")

def f(x):
	# var = 0.2
	# return 1.0/(1 + np.exp(-x*3))
	return 1.0/np.sqrt(2*np.pi) * np.exp(-x**2)

def plotter(predictor,label,trueFunc,useBlup):
	eps = predictor.TrueError(trueFunc)
	if useBlup:
		pt.plot(predictor.T,predictor.X_BLUP,label=f"BLUP, $\epsilon={predictor.blup_error:.3}$")
	pt.plot(predictor.T,predictor.X,label=label + f", $\epsilon={eps:.3}$")

#generate data
bottom = -4
top = 5
error_x = 0.05
data= pyclups.GenerateData(n=11,mode="uniform",xmax=top,xmin=bottom,noise=error_x,function=f,skedacity=0.2)
#specify predictor properties
K = pyclups.kernel.SquaredExponential(kernel_variance=0.5,kernel_scale=0.5)
basis = pyclups.basis.Hermite(5)
constraint = pyclups.constraint.Even()


#make predictions
tt = np.linspace(-5,5,201)+1
s = pyclups.Predictor(K,constraint,basis)
s.Verbose = True




#plot results
pt.errorbar(data.T,data.X,data.Errors,capsize=2,fmt='o',label="Sampled Data",zorder=1)
pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")

# for i in range(0,20,8):
pred = s.Predict(tt,data)
plotter(pred,f"CLUPS",f,True)
for i in [1000]:
	reg = pyclups.regularise.Curvature(i)
	pred = s.Predict(tt,data,reg)
	plotter(pred,f"rCLUPS_{i}",f,False)
pt.legend()
pt.ylabel("$z(t)$")
pt.xlabel("$t$")
pt.draw()
pt.pause(0.01)

# input("Enter to exit")s