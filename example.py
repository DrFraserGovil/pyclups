#!/opt/homebrew/bin/python3
import numpy as np
from matplotlib import pyplot as pt
from tqdm import tqdm
import warnings
from scipy import special
import pyclups



np.random.seed(0)
large_width = 400
np.set_printoptions(linewidth=large_width)
warnings.filterwarnings("ignore")

def f(x):
	var = 0.2
	return 1.0/(1 + np.exp(-x*3))

def plotter(predictor,label,trueFunc,useBlup):
	eps = predictor.TrueError(trueFunc)
	if useBlup:
		pt.plot(predictor.T,predictor.X_BLUP,label=f"BLUP, $\epsilon={predictor.blup_error:.3}$")
	pt.plot(predictor.T,predictor.X,label=label + f", $\epsilon={eps:.3}$")

#generate data
bottom = -5
top = 5
error_x = 0.05
data= pyclups.GenerateData(n=21,mode="uniform",xmax=top,xmin=bottom,noise=error_x,function=f,skedacity=0.5)
#specify predictor properties
K = pyclups.kernel.SquaredExponential(kernel_variance=0.5,kernel_scale=0.5)
basis = pyclups.basis.Hermite(3)
constraint = pyclups.constraint.Monotonic()


#make predictions
tt = np.linspace(bottom,top,111)
s = pyclups.Predictor(K,constraint,basis)
pred = s.Predict(tt,data)



#plot results
pt.errorbar(data.T,data.X,data.Errors,capsize=2,fmt='o',label="Sampled Data",zorder=1)
pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")

plotter(pred,"CLUPS",f,True)
pt.legend()
pt.ylabel("$z(t)$")
pt.xlabel("$t$")
pt.draw()
pt.pause(0.01)

input("Enter to exit")