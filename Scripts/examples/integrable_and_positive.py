import sys

sys.path.append('../')
import pyclups

import numpy as np
from matplotlib import pyplot as pt
import pyclups
pt.rcParams['text.usetex'] = True
np.random.seed(0)


def gauss(x,mu,var):
	return 1.0/np.sqrt(2*np.pi*var) * np.exp(-0.5*(x-mu)**2/var)
def f(x):
	var = 0.4
	x0 = -1
	x1 = 0
	w1 = 0
	w2 = 1

	# for i in 

	return w1*gauss(x,x0,var) + w2*gauss(x,x1,var)
	


bottom = -3
top = 3
t = []
x = []
# bounds = [[-3,-2.5],[-1.5,-0.5],[-0.5,0.5],[0.9,1.6],[1.7,3]]
# counts = [3,30,3,50,10]
# for i in range(len(counts)):
	
# 	[t1,x1] = pyclups.GenerateData(n=counts[i],mode="uniform",xmax=bounds[i][1],xmin=bounds[i][0],noise=0.05,function=f)
# 	# if counts[i] > 10:
# 	# 	x1  = x1*0.7
# 	# print(x1)
# 	t = np.concatenate((t,t1))
# 	x = np.concatenate((x,x1))
[t,x] = pyclups.GenerateData(n=8,mode="uniform",xmax=bottom,xmin=top,noise=0.03,function=f)
# x[np.abs(t)>0.4]-=0.03
error_x = 0.03

K = pyclups.kernel.SquaredExponential(kernel_variance=1.5,kernel_scale=0.8)
basis = pyclups.basis.Hermite(5)

tt = np.linspace(bottom,top,110)

c1 = pyclups.constraint.PositiveIntegrable(np.trapz(f(tt),tt))
c2 = pyclups.constraint.Positive()
c3 = pyclups.constraint.Integrable(np.trapz(f(tt),tt))

cons = [c2,c3,c1]
mode = ["Positive","Integrable","Positive\&Integrable"]
for i in range(len(cons)):
	s = pyclups.Predictor(K,cons[i],basis)
	pred = s.Predict(tt,t,x,error_x)
	eps = pred.TrueError(f)
	if i == 0:
		pt.plot(tt,f(tt),'k',linestyle='dotted',label="Real Function")
		pt.scatter(t,x,label="Sampled Data")
		pt.plot(pred.T,pred.X_BLUP,label=f"BLUP, $\epsilon={pred.blup_error:.2}$")
	pt.plot(pred.T,pred.X,label=f"{mode[i]}CLUPS, $\epsilon={eps:.2}$")
pt.legend()
pt.xlabel("$t$")
pt.ylabel("$z$")
pt.title("Positive \& Integrable Constraint")
pt.savefig("integrable_and_positive.pdf",format='pdf')

print("True",np.trapz(f(pred.T),pred.T))
print("blup",np.trapz(pred.X_BLUP,pred.T))
print("blup",np.trapz(pred.X,pred.T))