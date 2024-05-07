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

np.random.seed(4)
large_width = 400
np.set_printoptions(linewidth=large_width)
warnings.filterwarnings("ignore")

def f(x):
	# var = 0.2
	# return 1.0/(1 + np.exp(-x))
	return 1.0/np.sqrt(2*np.pi) * np.exp(-x**2/2)

def plotter(predictor,label,trueFunc,useBlup):
	eps = predictor.TrueError(trueFunc)
	if useBlup:
		pt.plot(predictor.T,predictor.X_BLUP,label=f"BLUP, $\epsilon={predictor.blup_error:.3}$",linestyle="dashdot")
	ls= "solid"
	if label=="rCLUPS":
		ls = "dashed"
	pt.plot(predictor.T,predictor.X,label=label + f", $\epsilon={eps:.3}$",linestyle=ls)
	pt.grid(True)
#generate data
bottom = -4
top = 4
error_x = 0.07
data= pyclups.GenerateData(n=11,mode="uniform",xmax=top,xmin=bottom,noise=error_x,function=f,skedacity=0.1)
#specify predictor properties
K = pyclups.kernel.SquaredExponential(kernel_variance=0.5,kernel_scale=1)
basis = pyclups.basis.Hermite(3)
# con2 = pyclups.constraint.Integrable(1)
constraint = pyclups.constraint.Unimodal()
# constraint.Add(pyclups.constraint.Positive(lambda t: t == -10))


#make predictions
tt = np.linspace(-4,4,121)
s = pyclups.Predictor(K,constraint,basis)
s.Verbose = True

# s2 = pyclups.Predictor(K,con2,basis)


#plot results
# fig,axs = pt.subplots(3,1)
pt.errorbar(data.T,data.X,data.Errors,color='k',capsize=2,fmt='o',label="_Sampled Data")
pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")

# for i in range(0,20,8):
pred = s.Predict(tt,data)
# pred2  = s2.Predict(tt,data)

# plotter(pred2,"CLUPS",f,True)
plotter(pred,f"CLUPS",f,True)
# print("rCLUPS")
# for i in [1]:
# 	qq = i /((tt[1] - tt[0])**2)
# 	reg = pyclups.regularise.Curvature(qq)
# 	pred = s.Predict(tt,data,reg)
# 	plotter(pred,f"rCLUPS",f,False)
pt.legend()
pt.ylabel("$z(t)$")
pt.xlabel("$t$")



# r = range(len(s.Convexity_Angles))
# axs[0].plot(r,s.Convexity_Angles)
# axs[0].set_xlabel("$\mathbf{w}$ index, $i$")
# axs[0].set_ylabel("Angle bettween of $\\frac{d\\mathbf{c}}{dw_i}$ and $\\frac{dL}{d\\mathbf{c}}$")
# axs[1].set_xlabel("$\mathbf{c}$ index, $j$")
# axs[1].set_ylabel("Component Norm")
# axs[0].grid()
# pt.xlabel("$t$")
# print(s.Convexity_DCDW.shape)
# axs[1].plot(s.Convexity_DCDW,label="$\\frac{d\\mathbf{c}}{dw_i}$")
# axs[1].plot(s.Convexity_dLdc,label="$\\frac{dL}{d\\mathbf{c}}$")
# axs[1].legend()
# axs[1].grid()
# axs[2].grid()
# print(s.Convexity_dLdc)
pt.draw()
pt.pause(0.01)
input("Enter to exit")