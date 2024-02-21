#!/opt/homebrew/bin/python3
import numpy as np
from matplotlib import pyplot as pt
from tqdm import tqdm
import warnings
from scipy import special
import pyclups



np.random.seed(3)
large_width = 400
np.set_printoptions(linewidth=large_width)
warnings.filterwarnings("ignore")

def f(x):
	var = 0.2
	# return 1.0/np.sqrt(2*np.pi*var) * np.exp(-0.5*x*x/var)

	return 1.0/(1 + np.exp(-x*3))+4


bottom = -3
top = 3
[t,x] = pyclups.GenerateData(n=10,mode="uniform",xmax=top,xmin=bottom,noise=0.1,function=f)
# t = [-3,0,3]
# x = [1.1,2,0.5]
error_x = 0.2

K = pyclups.kernel.SquaredExponential(kernel_variance=0.5,kernel_scale=0.5)
basis = pyclups.basis.Hermite(5,'even')

tt = np.linspace(bottom,top,200)
constraint = pyclups.constraint.Positive()
s = pyclups.Predictor(K,constraint,basis)





pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")

pt.scatter(t,x,label="Sampled Data")

# np.random.shuffle(tt)
pred = s.Predict(tt,t,x,error_x)
eps = pred.TrueError(f)
sorter = np.argsort(pred.T)



pt.plot(pred.T[sorter],pred.X_BLUP[sorter],label="BLUP $\epsilon=$"+str(pred.blup_error))
pt.plot(pred.T[sorter],pred.X[sorter],label="CLUP $\epsilon=$"+str(eps))

pt.legend()
pt.draw()
pt.pause(0.01)

input("Enter to exit")

# pt.cla()
# [a,b] = s.Angle
# pt.plot(a,b)
# pt.xlabel("Iterations")
# pt.ylabel("Mean cos(theta) between gradient components")
# pt.yscale('log')
# pt.draw()
# pt.pause(0.01)


