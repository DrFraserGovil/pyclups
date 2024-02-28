#!/opt/homebrew/bin/python3
import numpy as np
from matplotlib import pyplot as pt
from tqdm import tqdm
import warnings
from scipy import special
import pyclups



np.random.seed(1)
large_width = 400
np.set_printoptions(linewidth=large_width)
warnings.filterwarnings("ignore")

def f(x):
	var = 0.2
	# return 1.0/np.sqrt(2*np.pi*var) * np.exp(-0.5*x*x/var)

	return 1.0/(1 + np.exp(-x*3))

def plotter(x,y,axs,label):
	axs[0].plot(x,y,label=label)

	d = np.diff(y,1,0)/(x[1] - x[0])
	d2 = np.diff(y,2,0)/(x[1] - x[0])**2

	axs[1].plot(x[1:],d)
	axs[2].plot(x[2:],d2)
bottom = -3
top = 3
[t,x] = pyclups.GenerateData(n=11,mode="uniform",xmax=top,xmin=bottom,noise=0.05,function=f)
# t = [-3,0,3]
# x = [1.1,2,0.5]
error_x = 0.1

K = pyclups.kernel.SquaredExponential(kernel_variance=0.5,kernel_scale=0.5)
basis = pyclups.basis.Hermite(5)

tt = np.linspace(bottom,top,110)
# constraints = [pyclups.constraint.Monotonic(),pyclups.constraint.BoundedGradient(-0.1,1),pyclups.constraint.BoundedGradient(0,1),pyclups.constraint.BoundedGradient(0,0.75),pyclups.constraint.BoundedGradient(0,0.5)]
# names = ["Monotonic","(-0.1,1)-GradBound","(0,1)-GradBound","(0,0.75)-GradBound","(0,0.5)-GradBound"]
# constraints = [pyclups.constraint.Monotonic(),pyclups.constraint.Monotonic(),pyclups.constraint.PositiveBoundedGradient(0,1),pyclups.constraint.PositiveBoundedGradient(0,0.75),pyclups.constraint.PositiveBoundedGradient(0,0.5)]
# constraints[1].Add(pyclups.constraint.Positive(lambda t: t == tt[0]))
# names = ["Monotonic","Positive Monotonic","(0,1)-GradBound","(0,0.75)-GradBound","(0,0.5)-GradBound"]
constraints = [pyclups.constraint.Monotonic(),pyclups.constraint.MonotonicBoundedCurvature(-1,1),pyclups.constraint.MonotonicBoundedCurvature(-1,1)]
constraints[2].Add(pyclups.constraint.Positive(lambda t:t == tt[0]))
names = ["Monotonic","(-1,1)-MonoCurvBound","(-1,1)-+veMonoCurvBound"]
fig,axs = pt.subplots(3,1,figsize=(8,10))
axs[0].scatter(t,x,label="Sampled Data")
axs[0].plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")

for i in range(len(constraints)):
	print(names[i])
	s = pyclups.Predictor(K,constraints[i],basis)


	pred = s.Predict(tt,t,x,error_x)
	eps = pred.TrueError(f)
	sorter = np.argsort(pred.T)


	if i == 0:
		plotter(pred.T[sorter],pred.X_BLUP[sorter],axs,f"BLUP $\epsilon={pred.blup_error:.3}$")
	plotter(pred.T[sorter],pred.X[sorter],axs,f"{names[i]} $\epsilon={eps:.3}$")

axs[0].legend()
axs[0].set_ylabel("$z(t)$")
axs[1].set_ylabel("$z^\prime(t)$")
axs[2].set_ylabel("$z^{\prime\prime}(t)$")
axs[2].set_xlabel("$t$")
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


