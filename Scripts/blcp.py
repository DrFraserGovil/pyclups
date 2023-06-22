#!/usr/bin/python3


import numpy as np
import imageio
import matplotlib.colors as colors
from matplotlib import pyplot as pt
from tqdm import tqdm
# np.random.seed(0) #enable for reproducable randomness
import warnings
import time
large_width = 400
np.set_printoptions(linewidth=large_width)
warnings.filterwarnings("ignore")
kernelSigma = 3.5

dataNoise = 0.00001
learningRate = 0.1
learningMemory = 0.8
learningMemory_SecondMoment = 0.99

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
def strRound(q):
	return '%s' % float('%.3g' % q)
def BLP(predictT,dataT,dataX):
	muData = Prior(dataT)#np.mean(dataX)
	
	#precompute some useful quantities
	K=kernelMatrix(dataT) + (dataNoise/2)**2 * np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	KinvX = np.matmul(Kinv,dataX-muData)

	#loop over the prediction points, and compute the prediction at each one
	mean = Prior(predictT)
	ps = np.zeros(len(predictT))
	rms = 0
	trueY = Func(predictT)
	for i in range(len(predictT)):
		t = predictT[i]
		
		k=kernelVector(dataT,t)
		ps[i] = np.dot(KinvX,k) + mean[i] #normal BLP

		rms += (trueY[i] - ps[i])**2
		# print(predictX[i],ps[i],rms)
	# ps += meanFunc(predictX)
	rms = np.sqrt(rms/len(ps))
	return [ps,rms]



def C_BLP(predictX,dataT,dataX,steps):
	gPredict = Prior(predictX)
	trueY = Func(predictX)
	tData = dataX - Prior(dataT)
	K=kernelMatrix(dataT) + (dataNoise/2)**2 * np.identity(len(dataT)) #softening *kernel(0,0)* np.identity(len(dataT))
	Kinv = np.linalg.inv(K)

	q = np.zeros((len(predictX),1))
	for i in range(len(predictX)):
		t = predictX[i]
		
		k=kernelVector(dataT,t)
		v = np.matmul(Kinv,k)
		q[i] = np.dot(v,tData) + gPredict[i]

	mDim = len(predictX) -1
	zs = np.zeros((mDim,1))-7
	cs = np.exp(zs)
	D = np.zeros((mDim,mDim+1))
	for i in range(mDim):
		D[i,i] = -1
		D[i,i+1] = 1
	
	R = np.matmul(D.transpose(), np.linalg.inv(np.matmul(D, D.transpose())))
	Rt = R.transpose()
	Rdq = np.matmul(np.matmul(R,D),q)
	J = np.matmul(np.identity(len(predictX)) - np.matmul(R,D),q)

	ms = np.zeros((len(zs),1))
	vs = np.zeros((len(zs),1))
	grad = np.zeros((len(zs),1))
	for s in range(steps):
		
		cs = np.exp(zs)

		Rc = np.matmul(R,cs)
		# predict = J + Rc
		diff = Rc - Rdq
		# for i in range(len(grad)):
		# 	# col = R[:,i] * cs[i]
		# 	g = 0
		# 	for j in range(len(grad)):
		# 		g+= diff[j] * R[j,i] * cs[i] #col[j]
		# 	grad[i] = g
		# # Rtdiff = np.matmul(Rt,diff)
		# print(np.shape(diff),np.shape(cs))
		# grad2 = np.matmul(cs.transpose(),np.matmul(Rt,diff))
		grad = np.multiply(cs,np.matmul(Rt,diff))
		# print(grad.transpose()[1:5],grad2.transpose()[1;5÷])
		# print(zs.transpose(),"\n",cs.transpose(),"\n",predict.transpose(),"\n",diff.transpose(),"\n",grad.transpose(),"\n\n")
		# print("grad=",grad)
		#ADAM step routine
		ms = learningMemory * ms + (1.0 - learningMemory)*grad
		vs = learningMemory_SecondMoment * vs + (1.0 - learningMemory_SecondMoment)*np.multiply(grad,grad)
		c1 = 1.0 - learningMemory**(s+1)
		c2 = 1.0 - learningMemory_SecondMoment**(s+1)
		eps = 1e-8
		# r = 
		# print(r)
		# print(np.shape(zs),np.shape(ms),np.shape(vs),np.shape(r))
		zs -= learningRate * np.divide(ms/c1,np.sqrt(eps + vs/c2))
		# zs -= learningRate * grad
		#prevents the prediction from 'dying' by going too negative
		for j in range(1,len(zs)):
			m = -20
			if zs[j] < m:
				zs[j] = m
			l = 20
			if zs[j] > l:
				zs[j] = l


	bestPredict = J + np.matmul(R,cs)
	rms = 0
	for i in range(len(ps)):
		rms += (trueY[i] - bestPredict[i])**2
	rms/=len(trueY)
	# print(cs.transpose())
	# print(bestPredict.transpose())
	return [bestPredict,np.sqrt(rms)]
def BLCP(predictX,dataT,dataX,steps,zs	):
	# zs = np.zeros(len(predictX))
	# zs[1:]= -2
	ms = np.zeros(len(zs))
	vs = np.zeros(len(zs))
	trueY = Func(predictX)
	gPredict = Prior(predictX)
	tData = dataX - Prior(dataT)
	#precompute values as before
	K=kernelMatrix(dataT) + (dataNoise/2)**2 * np.identity(len(dataT)) #softening *kernel(0,0)* np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	mu = 0# np.mean(dataX)
	w = np.matmul(Kinv,tData)

	#now precompute some values which vary with time, so store them in vectors
	#also compute the "stupid" value as a base case for initialisation
	kdotw = np.zeros(len(predictX))
	As = np.zeros(len(predictX))
	ks = [np.zeros(len(dataT))]*len(predictX)
	for i in range(len(predictX)):
		t = predictX[i]
		
		k=kernelVector(dataT,t)
		ks[i] = k
		v = np.matmul(Kinv,k)
		As[i] = np.dot(v,tData)
		kdotw[i] = np.dot(k,w)

	Q = As + gPredict
	T = Transform(zs)
	#iterate over a number of steps, computing the optimisation step per the ADAM optimiser
	#learning rate and memory parameters defined globally for ease

	#generate empty gradient vector
	grad = np.zeros(len(zs))
	for s in range(steps):
		# print(s)
		T = Transform(zs)
		# if s == steps -1:
		# 	print("final pos",T,"\n",zs)
		for j in range(len(zs)):
			dTdz_j = TransformDerivative(zs,j)
			# print("grad",grad,"\n","pos",T)
			# grad[j] += 2 * (T[j] - Q[j])
			g = 0
			for i in range(len(T)):
				g += 2 * (T[i] - Q[i]) * dTdz_j[i]
				# if s == steps -1:
				# 	print("\t",j,(T[i] - Q[i]),dTdz_j[i],g)
			grad[j] = g
		# print("grad=",grad)
		#ADAM step routine
		ms = learningMemory * ms + (1.0 - learningMemory)*grad
		vs = learningMemory_SecondMoment * vs + (1.0 - learningMemory_SecondMoment)*np.multiply(grad,grad)
		c1 = 1.0 - learningMemory**(s+1)
		c2 = 1.0 - learningMemory_SecondMoment**(s+1)
		eps = 1e-8
		zs -= learningRate * np.divide(ms/c1,np.sqrt(eps + vs/c2))
		# zs -= learningRate * grad
		#prevents the prediction from 'dying' by going too negative
		for j in range(1,len(zs)):
			m = -20
			if zs[j] < m:
				zs[j] = m
			l = 20
			if zs[j] > l:
				zs[j] = l

	

	ps = Transform(zs) + mu
	# for iT in range(0,len(predictX)):
	# 	k = ks[iT]
	# 	v = np.matmul(Kinv,k)
	# 	ait = v + (ps[iT] - As[iT] - gPredict[iT])/(kdotw[iT])
		
	# 	print("At i=",iT,"MSE=",np.dot(ait,np.matmul(K,ait)) -2 *np.dot(ait,tData))
	
	rms = 0
	for i in range(len(ps)):
		rms += (trueY[i] - ps[i])**2
	rms/=len(trueY)
	# ps += meanFunc(predictX)
	return [ps,np.sqrt(rms)]


mode = 0

if mode == 0:
	def Transform(z):
		# return z
		out = np.zeros(np.shape(z))
		out[0] = z[0]
		for i in range(1,len(z)):
			out[i]=out[i-1] + np.exp(z[i])
		# print("hi")
		return out

	def TransformDerivative(z,i):
		val = 1
		if i > 0:
			val =np.exp(z[i])

		out = np.zeros(np.shape(z))
		out[i:] = val
		return out
	def Func(t):
		return 1.0/(1 + np.exp(-t)) + 1.0/(1 + np.exp(-3*(t-7))) 
if mode == 1:
	deltaT = 1
	def Transform(z):
		out = np.exp(z)
		out /= (deltaT * np.sum(out))
		return out

	def TransformDerivative(z,i):
		T = Transform(z)
		q = -T*deltaT
		q[i] += 1
		return q* T[i]
		
	def Func(t):
		return 1.0/(np.sqrt(2*np.pi)) * np.exp(- (t)**2/2)
if mode == 2:
	alpha = 0.5
	def Transform(z):
		li = np.zeros(np.shape(z))
		li[0] = z[0]
		li[1:] = alpha * (np.exp(z[1:])-1)/(np.exp(z[1:])+1)

		out = np.cumsum(li)
		# print(z)
		# print(li)
		# print(np.cumsum(li))
		# print(out)
		# p = o
		return np.exp(out)
	def TransformDerivative(z,i):
		# li = np.zeros(np.shape(z))
		# li[0] = z[0]
		# li[1:] = alpha * (np.exp(z[1:])-1)/(np.exp(z[1:])+1)

		T= Transform(z)
		g = np.zeros(np.shape(z))
		if i == 0:
			g += T
		else:
			
			g[i:] = 2*alpha*T[i:]*np.exp(-z[i])/(np.exp(-z[i])+1)**2
		# print("i=",i)
		# print(z)
		# print(T)
		# print(g)
	
		return g
		
	def Func(t):
		sig = 1.5
		return 1.0/(np.sqrt(2*np.pi)*sig) * np.exp(- (t)**2/(2*sig**2))
xMin = -6
xMax = 5
m = (Func(xMax) - Func(xMin))/(xMax - xMin)
c = Func(xMin) -xMin *m
def Prior(t):
	return m * t + c
	# return Func(t)
	# return t-t
def GenerateData(nData):
	# t = np.linspace(xMin,xMax,nData) + np.random.normal(0,1,nData,)
	t = np.random.uniform(xMin,xMax,nData)
	x = Func(t) * (1 + np.random.normal(0,dataNoise,nData,))+np.random.normal(0,0.01,nData,)

	return [t,x]
np.random.seed(0)
[t,x] = GenerateData(25)
c = np.mean(x)
tt = np.linspace(min(t),max(t),1000)
pt.plot(tt,Func(tt),"k:",label="Underlying function")
pt.plot(tt,Prior(tt),"r:",label="Prior")
tt = np.linspace(min(t),max(t),300)
deltaT = tt[1] - tt[0]
pt.scatter(t,x,label="Data")


[ps,rms] = BLP(tt,t,x)
pt.plot(tt,ps,label="BLP, $\epsilon=$" + strRound(rms))



# print(np.sum(ps) * deltaT)

# zinit = ps.copy()
# prev = zinit[0]
# for i in range(1,len(ps)):
# 	if ps[i] > prev:
# 		zinit[i] = np.log(ps[i] - prev)
# 		prev = ps[i]
# 	else:
# 		zinit[i] = -10

zinit = np.zeros(np.shape(ps))-5
zinit[0] = -2#np.log(np.maximum(ps[0],0.01))

Res = 400
pt.title(str(Res) + " optim loops")
alpha= 0.2*deltaT
start = time.time()
[ps,rms] = BLCP(tt,t,x,Res,zinit.copy())
end = time.time()
pt.plot(tt,ps,label="Old, $\\tau=$" + strRound(end-start) + "s")
print("Old: ",end-start)
start = time.time()
[ps,rms] = C_BLP(tt,t,x,Res)
end = time.time()
print("New: ",end-start)
pt.plot(tt,ps,label="New, $\\tau=$" + strRound(end-start)+"s")


start = time.time()
[ps,rms] = C_BLP(tt,t,x,10000)
end = time.time()
print("Hires: ",end-start)
pt.plot(tt,ps,label="New @ 10,000res, $\\tau=$" + strRound(end-start) + "s")
# [ps,rms] = C_BLP(tt,t,x,1200)
# pt.plot(tt,ps,label="C-BLP-hi, $\epsilon=$" + strRound(rms))
# [ps,rms] = BLCP(tt,t,x,3000,zinit.copy())
# print(np.sum(ps) * deltaT)
# pt.plot(tt,ps,label="BLCP-0.1-hi, $\epsilon=$" + strRound(rms))

# [ps,rms] = BLCP(tt,t,x,1500,zinit)
# print(np.sum(ps) * deltaT)
# pt.plot(tt,ps,label="BLCP-hi, $\epsilon=$" + strRound(rms))

pt.legend()
pt.draw()
pt.pause(0.01)

input("Enter to exit")