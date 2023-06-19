#!/usr/bin/python3
import numpy as np
import imageio
import matplotlib.colors as colors
from matplotlib import pyplot as pt
from tqdm import tqdm
# np.random.seed(0) #enable for reproducable randomness
import warnings
warnings.filterwarnings("ignore")
nData = 150
nPredict = 100
dataNoise = 5e-3
kernelSigma = 0.2
softening = 1e-2
learningRate = 0.1
learningMemory = 0.9
learningMemory_SecondMoment = 0.999
useFlatStart = True
maxOptimSteps = 1000
stepDelta = 1
m = 0
c=0
def strRound(q):
	return '%s' % float('%.3g' % q)
def meanFunc(x):
	return m*x+c

def kernel(x,y):
	#covariance the kernel
	d = abs(x-y)/kernelSigma
	# return dataNoise*dataNoise/(1+(d)**2)
	return  (np.exp(-0.5 * d**2))
def kernelMatrix(sampleX):
	#my attempt at computing K_ij, the covariance/ second moment matrix evaluated over the data
	n = len(sampleX)
	K = np.zeros(shape=(n,n))
	for i in range(n):
		for j in range(n):
			K[i,j] = kernel(sampleX[i],sampleX[j])
	return K

def kernelVector(sampleX,t):
	#my attempt at computing k_i, the covariance/ second moment vector evaluated over the data, at time t
	n = len(sampleX)
	k = np.zeros(shape=(n,))
	for i in range(n):
		k[i] = kernel(sampleX[i],t)
	return k

def naiveScore(predictX,dataT,dataX,trueY):
	#standard BLP mechanism
	muData = meanFunc(dataT)#np.mean(dataX)
	
	#precompute some useful quantities
	K=kernelMatrix(dataT) + (dataNoise * dataNoise) * np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	KinvX = np.matmul(Kinv,dataX-muData)

	#loop over the prediction points, and compute the prediction at each one
	mean = meanFunc(predictX)
	ps = np.zeros(len(predictX))
	rms = 0
	for i in range(len(predictX)):
		t = predictX[i]
		
		k=kernelVector(dataT,t)
		ps[i] = np.dot(KinvX,k) + mean[i] #normal BLP

		rms += (trueY[i] - ps[i])**2
		# print(predictX[i],ps[i],rms)
	# ps += meanFunc(predictX)
	rms = np.sqrt(rms/len(ps))
	return [ps,rms]


def globalScore(predictX,dataT,dataX,steps,trueY):
	zs = np.zeros(len(predictX))
	zs[1:]=-10
	ms = np.zeros(len(zs))
	vs = np.zeros(len(zs))
	gPredict = meanFunc(predictX)
	tData = dataX - meanFunc(dataT)
	#precompute values as before
	K=kernelMatrix(dataT) + (dataNoise * dataNoise) * np.identity(len(dataT)) #softening *kernel(0,0)* np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	mu = 0# np.mean(dataX)
	w = np.matmul(Kinv,tData)

	#now precompute some values which vary with time, so store them in vectors
	#also compute the "stupid" value as a base case for initialisation
	kdotw = np.zeros(len(predictX))
	As = np.zeros(len(predictX))
	ks = [np.zeros(len(dataT))]*len(predictX)
	prev = 0
	for i in range(len(predictX)):
		t = predictX[i]
		
		k=kernelVector(dataT,t)
		ks[i] = k
		v = np.matmul(Kinv,k)
		As[i] = np.dot(v,tData)
		kdotw[i] = np.dot(k,w)

		test = np.dot(w,k) + gPredict[i]
		if i == 0:
			zs[i] = test
			prev = test
		else:
			if test > prev:
				zs[i] = np.log(test-prev)
				prev = test
			else:
				zs[i] = -10

	#iterate over a number of steps, computing the optimisation step per the ADAM optimiser
	#learning rate and memory parameters defined globally for ease

	#generate empty gradient vector
	grad = np.zeros(len(zs))
	for s in range(steps):

		

		#convert zs into prediction values
		deltas = np.append(zs[0],np.exp(zs[1:])) 
		S = np.cumsum(deltas) 
		#this is my attempt to efficiently compute the gradient calculations in Eq. 31
		bracket = 2*(S - (As + gPredict))
		bSum = np.sum(bracket)
		grad[0] = np.sum(bracket)
		for j in range(1,len(bracket)):
			grad[j] = grad[j-1] - bracket[j-1]
		for j in range(1,len(bracket)):
			grad[j] *= deltas[j]
		#ADAM step routine
		ms = learningMemory * ms + (1.0 - learningMemory)*grad
		vs = learningMemory_SecondMoment * vs + (1.0 - learningMemory_SecondMoment)*np.multiply(grad,grad)
		c1 = 1.0 - learningMemory**(s+1)
		c2 = 1.0 - learningMemory_SecondMoment**(s+1)
		eps = 1e-8
		zs -= learningRate * np.divide(ms/c1,np.sqrt(eps + vs/c2))

		#prevents the prediction from 'dying' by going too negative
		for j in range(1,len(zs)):
			m = -20
			if zs[j] < m:
				zs[j] = m

	ps = np.zeros(np.shape(zs))
	rms = 0
	for i in range(len(ps)):
		if i > 0:
			ps[i] = ps[i-1] + np.exp(zs[i])
		else:
			ps[i] = zs[i] + mu
		rms += (trueY[i] - ps[i])**2
	rms/=len(trueY)
	# ps += meanFunc(predictX)
	return [ps,np.sqrt(rms)]

# dataX = np.array([0,5,10])
# dataY = np.array([3,6,10])

def plot(dataX,dataY,sigma,priorM,priorC,ax,func):
	global m, c
	m = priorM
	c = priorC
	global kernelSigma
	kernelSigma = sigma
	sampleX = np.linspace(np.min(dataX),np.max(dataX),1000)
	tY = func(sampleX)
	[ps,rms] = naiveScore(sampleX,dataX,dataY,tY)
	[ps2,rms2] = globalScore(sampleX,dataX,dataY,500,tY)
	ax.plot(sampleX,tY,'k:',label="True Function")
	ax.plot(sampleX,meanFunc(sampleX),':',label="Prior")
	ax.scatter(dataX,dataY,label="Observed Data")
	ax.plot(sampleX,ps,label="BLP $\epsilon = " + strRound(rms) +"$")
	ax.plot(sampleX,ps2,label="BLMP $\epsilon = " + strRound(rms2) +"$")
	# pt.plot(sampleX,psHalf,label="BLP $\sigma = 0.5$")
	ax.set_xlabel("$t$")
	ax.set_ylabel("$X_t$")
	ax.legend()
def logit(x):
	return 1.0/(1 + np.exp(-x))
def func(x):
	return logit(3*x) + 2*logit((x-4)/2) 
# np.random.seed(1)x
d = 10
dataX = np.linspace(-5,10,d) 
dataX += np.random.normal(0,0.5*np.ptp(dataX)/d,d)
dataX = np.sort(dataX)
dataY = func(dataX)+ np.random.normal(0,0.05,d)
fig,axs = pt.subplots(2,3)
sig = 1
axs[0][0].set_title("Normal BLP")
plot(dataX,dataY,sig,0,0,axs[0][0],func)

axs[1][0].set_title("Mean-Scaled BLP")
m = 0
qX = dataX[1:-1]
qY = dataY[1:-1]
c = np.mean([dataY[0],dataY[-1]])
plot(qX,qY,sig,0,c,axs[1][0],func)

axs[0][1].set_title("Linear Prior BLP")
m = (dataY[-1] - dataY[0])/(dataX[-1] - dataX[0])
c = dataY[0] - m * dataX[0]

plot(qX,qY,sig,m,c,axs[0][1],func)

axs[1][1].set_title("Exact Prior")

def meanFunc(x):
	return func(x)

plot(dataX,dataY,sig,m,c,axs[1][1],func)


axs[0][2].set_title("Bad Prior")
def meanFunc(x):
	return 5-(x/5)**2
plot(dataX,dataY,sig,m,c,axs[0][2],func)
axs[0][2].set_ylim(-1,5)
axs[1][2].set_title("Stupid Prior")
def meanFunc(x):
	return np.sin(5*x)+3 -np.abs(x-1)**0.5
plot(dataX,dataY,sig,m,c,axs[1][2],func)
# pt.plot(sampleX,pred)

pt.draw()

pt.pause(0.1)
input("Enter to exit")
