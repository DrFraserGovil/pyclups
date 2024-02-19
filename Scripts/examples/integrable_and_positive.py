import sys

sys.path.append('../')
import pyclup

import numpy as np
from matplotlib import pyplot as pt
import pyclup
pt.rcParams['text.usetex'] = True
np.random.seed(1)


def gauss(x,mu,var):
	return 1.0/np.sqrt(2*np.pi*var) * np.exp(-0.5*(x-mu)**2/var)
def f(x):
	var = 1
	x0 = -1
	x1 = 0
	w1 = 0
	w2 = 1

	# for i in 

	return w1*gauss(x,x0,var) + w2*gauss(x,x1,var)+(1-w2-w1)/6
	


bottom = -3
top = 3
t = []
x = []
# bounds = [[-3,-2.5],[-1.5,-0.5],[-0.5,0.5],[0.9,1.6],[1.7,3]]
# counts = [3,30,3,50,10]
# for i in range(len(counts)):
	
# 	[t1,x1] = pyclup.GenerateData(n=counts[i],mode="uniform",xmax=bounds[i][1],xmin=bounds[i][0],noise=0.05,function=f)
# 	# if counts[i] > 10:
# 	# 	x1  = x1*0.7
# 	# print(x1)
# 	t = np.concatenate((t,t1))
# 	x = np.concatenate((x,x1))
[t,x] = pyclup.GenerateData(n=20,mode="semi",xmax=bottom,xmin=top,noise=0.05,function=f)
# x[np.abs(t-1)>0.4]-=0.1
error_x = 0.5

K = pyclup.kernel.SquaredExponential(kernel_variance=0.5,kernel_scale=0.5)
basis = pyclup.basis.Hermite(7)

tt = np.linspace(bottom,top,310)

constraint = pyclup.constraint.PositiveIntegrable(np.trapz(f(tt),tt))
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
pt.savefig("integrable_and_positive.pdf",format='pdf')

print("True",np.trapz(f(pred.T),pred.T))
print("blup",np.trapz(pred.X_BLUP,pred.T))
print("blup",np.trapz(pred.X,pred.T))