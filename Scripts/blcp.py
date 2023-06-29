#!/usr/bin/python3


import numpy as np
from matplotlib import pyplot as pt
from tqdm import tqdm
import warnings
from scipy import special
large_width = 400
np.set_printoptions(linewidth=large_width)
warnings.filterwarnings("ignore")
kernelSigma = 3.5

dataNoise = 2
learningRate = 0.05
learningMemory = 0.7
learningMemory_SecondMoment = 0.99

def kernel(x,y):
	#covariance the kernel
	d = abs(x-y)/kernelSigma
	# return dataNoise*dataNoise/(1+(d)**2)
	return 30*(np.exp(-0.5 * d**2))
def kernelMatrix(sampleX):
	#my attempt at computing K_ij, the covariance/ second moment matrix evaluated over the data
	n = len(sampleX)
	K = np.zeros(shape=(n,n))
	for i in range(n):
		for j in range(n):
			K[i,j] = kernel(sampleX[i],sampleX[j])
	return K


def phi(i,t):
	p_monic = special.hermite(i, monic=True)
	return p_monic(t)
	
def kernelVector(sampleX,t):
	#my attempt at computing k_i, the covariance/ second moment vector evaluated over the data, at time t
	n = len(sampleX)
	k = np.zeros(shape=(n,))
	for i in range(n):
		k[i] = kernel(sampleX[i],t)
	return k
def strRound(q):
	return '%s' % float('%.7g' % q)
def BLP(predictT,dataT,dataX):
	muData = Prior(dataT)#np.mean(dataX)
	
	#precompute some useful quantities
	K=kernelMatrix(dataT) +  dataNoise*dataNoise* np.identity(len(dataT))
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
	# ps += meanFunc(predictX)
	rms = np.sqrt(rms/len(ps))
	return [ps,rms]



def C_BLP(predictX,dataT,dataX,steps):
	gPredict = Prior(predictX)
	trueY = Func(predictX)
	tData = dataX - Prior(dataT)
	K=kernelMatrix(dataT) +  dataNoise*dataNoise*np.identity(len(dataT)) #softening *kernel(0,0)* np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	w = np.matmul(Kinv,tData)
	B = np.dot(w,tData)
	q = np.zeros((len(predictX),1))
	v = []
	for i in range(len(predictX)):
		t = predictX[i]
		
		k=kernelVector(dataT,t)
		vi = np.matmul(Kinv,k)
		v.append(vi)
		q[i] = np.dot(vi,tData) + gPredict[i]
	# return [q,0,0]
	mDim = len(predictX) -1
	# drange = np.ptp(dataX)/mDim
	zs = np.random.uniform(-6,-3,(mDim,1))
	prev = q[0]
	for i in range(1,len(predictX)):
		pred = q[i]
		naivediff = pred - prev
		if naivediff > 0:
			zs[i-1] = np.log(naivediff)
			prev = pred
		else:
			zs[i-1] = -4

	cs = np.exp(zs)
	D = np.zeros((mDim,mDim+1))
	for i in range(mDim):
		D[i,i] = -1
		D[i,i+1] = 1
	DDtinv = np.linalg.inv(np.matmul(D, D.transpose()))
	R = np.matmul(D.transpose(), DDtinv)
	Rt = R.transpose()
	Rdq = np.matmul(np.matmul(R,D),q)
	J = np.matmul(np.identity(len(predictX)) - np.matmul(R,D),q)

	ms = np.zeros((len(zs),1))
	vs = np.zeros((len(zs),1))
	grad = np.zeros((len(zs),1))
	for s in range(steps):
		
		cs = np.exp(zs)

		Rc = np.matmul(R,cs)
		diff = Rc - Rdq
		grad = np.multiply(cs,np.matmul(Rt,diff))
		
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
			l = 20
			if zs[j] > l:
				zs[j] = l
	cs = np.exp(zs)
	bestRc = np.matmul(R,cs)
	bestPredict = J + bestRc
	bestEta =1.0/B * np.matmul(R, np.matmul(D,q) - cs)
	rms = 0
	mse = 0
	for i in range(len(bestPredict)):
		rms += (trueY[i] - bestPredict[i])**2
		
		t = predictX[i]
		bestA = v[i] - bestEta[i] * w
		k=kernelVector(dataT,t)
		contrib = np.matmul(bestA.transpose(),np.matmul(K,bestA)) - 2*np.matmul(bestA.transpose(),k)
		mse += contrib
	mse/=len(trueY)
	rms/=len(trueY)
	return [bestPredict,np.sqrt(rms),mse]
def BLCP(predictX,dataT,dataX,steps,zs	):
	# zs = np.zeros(len(predictX))
	# zs[1:]= -2
	ms = np.zeros(len(zs))
	vs = np.zeros(len(zs))
	trueY = Func(predictX)
	gPredict = Prior(predictX)
	tData = dataX - Prior(dataT)
	#precompute values as before
	K=kernelMatrix(dataT) + (dataNoise)**2 * np.identity(len(dataT)) #softening *kernel(0,0)* np.identity(len(dataT))
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
		T = Transform(zs)
		for j in range(len(zs)):
			dTdz_j = TransformDerivative(zs,j)
			g = 0
			for i in range(len(T)):
				g += 2 * (T[i] - Q[i]) * dTdz_j[i]
			grad[j] = g
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
	rms = 0
	for i in range(len(ps)):
		rms += (trueY[i] - ps[i])**2
	rms/=len(trueY)
	# ps += meanFunc(predictX)
	return [ps,np.sqrt(rms)]

def BLUP(predictX,dataT,dataX,order):
	
	#precompute some useful quantities
	K=kernelMatrix(dataT) +  dataNoise*dataNoise*np.identity(len(dataT)) 
	Kinv = np.linalg.inv(K)
	Phi = np.zeros((order+1,len(dataT)))
	for m in range(0,order+1):
		for j in range(len(dataT)):
			Phi[m,j] = phi(m,dataT[j])

	PhiT = Phi.transpose()
	M = np.matmul(Phi,np.matmul(Kinv,PhiT))
	Minv = np.linalg.inv(M)
	obj = np.matmul(PhiT,Minv)
	n = len(dataT)
	mat1 = np.matmul(Kinv,np.identity(n)-np.matmul(obj,np.matmul(Phi,Kinv)))

	ps = np.zeros(len(predictX),)
	phiVec = np.zeros(order+1)
	trueY = Func(predictX)
	rms = 0
	
	for i in range(len(predictX)):
		t = predictX[i]
		
		for j in range(order+1):
			phiVec[j] = phi(j,t)

		
		k = kernelVector(dataT,t)
		


		a = np.matmul(mat1,k) + np.matmul(np.matmul(Kinv,obj),phiVec)
		
		ps[i] = np.dot(a,dataX)
		rms += (trueY[i] - ps[i])**2

	rms = np.sqrt(rms/len(ps))
	return [ps,rms]


def CLUP(predictX,dataT,dataX,order,steps):

	trueY = Func(predictX)
	K=kernelMatrix(dataT) +  dataNoise*dataNoise*np.identity(len(dataT)) #softening *kernel(0,0)* np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	w = np.matmul(Kinv,dataX)
	q = np.zeros((len(predictX),1))
	n = len(dataX)
	Phi = np.zeros((order+1,len(dataT)))
	for m in range(0,order+1):
		for j in range(len(dataT)):
			Phi[m,j] = phi(m,dataT[j])
	PhiT = Phi.transpose()
	phis = []
	v = []
	M = np.matmul(Phi,np.matmul(Kinv,PhiT))
	Minv = np.linalg.inv(M)
	C = np.matmul(np.matmul(Kinv,PhiT),Minv)
	Bmat = np.identity(n)-np.matmul(C,Phi)
	alpha = np.zeros((len(predictX),1))
	beta = np.zeros((len(predictX),1))
	curlyB = np.zeros((len(beta),len(beta)))
	ell = np.zeros((len(predictX),1))
	vecs =[]
	ks = []
	ellBottom = np.dot(w,np.matmul(Bmat,np.matmul(K,np.matmul(Bmat,w))))
	ellLeft = np.matmul(K,np.matmul(Bmat,w))
	for i in range(len(predictX)):
		# print(i)
		t = predictX[i]
		
		k=kernelVector(dataT,t)
		ks.append(k)
		vi = np.matmul(Kinv,k)
		v.append(vi)
		phiVec = np.zeros(order+1)
		for j in range(order+1):
			phiVec[j] = phi(j,t)
		phis.append(phiVec)

		vec = np.matmul(Bmat,vi) + np.matmul(C,phiVec)
		vecs.append(vec)
		alpha[i] = np.dot(vec,dataX)
		beta[i] = np.dot(np.matmul(Bmat,w),dataX)
	
		curlyB[i,i] = beta[i]

		ellTop = np.dot(ellLeft,vec) - np.dot(k,np.matmul(Bmat,w))
		ell[i] = ellTop/ellBottom
		


	mDim = len(predictX) -1
	zs = np.random.uniform(-5,-2,(mDim,1))
	prev = np.dot(vecs[0],dataX)
	for i in range(1,len(predictX)):
		pred = np.dot(vecs[i],dataX)
		naivediff = pred - prev
		if naivediff > 0:
			zs[i-1] = np.log(naivediff)
			prev = pred
		else:
			zs[i-1] = -4

	cs = np.exp(zs)
	D = np.zeros((mDim,mDim+1))
	for i in range(mDim):
		D[i,i] = -1
		D[i,i+1] = 1


	DBDTinv = np.linalg.inv(np.matmul(D,np.matmul(curlyB,D.transpose())))
	Dalpha = np.matmul(D,alpha)
	H = np.matmul(D.transpose(),DBDTinv)
	Ht = H.transpose()
	ms = np.zeros((len(zs),1))
	vs = np.zeros((len(zs),1))
	grad = np.zeros((len(zs),1))
	for s in range(steps):
		cs = np.exp(zs)

		diff = np.matmul(H,cs-Dalpha)+ell
		
		grad = np.multiply(cs,np.matmul(Ht,diff))
		#ADAM step routine
		ms = learningMemory * ms + (1.0 - learningMemory)*grad
		vs = learningMemory_SecondMoment * vs + (1.0 - learningMemory_SecondMoment)*np.multiply(grad,grad)
		c1 = 1.0 - learningMemory**(s+1)
		c2 = 1.0 - learningMemory_SecondMoment**(s+1)
		eps = 1e-8
		zs -= learningRate * np.divide(ms/c1,np.sqrt(eps + vs/c2))
		#prevents the prediction from 'dying' by going too negative
		for j in range(1,len(zs)):
			m = -10
			if zs[j] < m:
				zs[j] = m
			l = 20
			if zs[j] > l:
				zs[j] = l
	cs= np.exp(zs)
	ps = np.zeros(len(predictX),)
	rms=0
	correct = np.matmul(H,cs-Dalpha)
	R = np.matmul(Bmat,w)
	mse = 0
	for i in range(len(predictX)):

		ai = vecs[i] + correct[i] * R
		mse += np.dot(ai,K@ai) - 2 * np.dot(ai,ks[i])
		ps[i] = np.dot(ai,dataX)
		rms += (trueY[i] - ps[i])**2
	rms = np.sqrt(rms/len(ps))
	return [ps,rms,mse]

def CLUP_Boosted(predictX,dataT,dataX,order,steps):

	trueY = Func(predictX)
	K=kernelMatrix(dataT) +  dataNoise*dataNoise*np.identity(len(dataT)) #softening *kernel(0,0)* np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	w = np.matmul(Kinv,dataX)
	q = np.zeros((len(predictX),1))
	n = len(dataX)
	Phi = np.zeros((order+1,len(dataT)))
	for m in range(0,order+1):
		for j in range(len(dataT)):
			Phi[m,j] = phi(m,dataT[j])
	PhiT = Phi.transpose()
	phis = []
	v = []
	M = np.matmul(Phi,np.matmul(Kinv,PhiT))
	Minv = np.linalg.inv(M)
	C = np.matmul(np.matmul(Kinv,PhiT),Minv)
	Bmat = np.identity(n)-np.matmul(C,Phi)
	alpha = np.zeros((len(predictX),1))
	beta = np.zeros((len(predictX),1))
	curlyB = np.zeros((len(beta),len(beta)))
	ell = np.zeros((len(predictX),1))
	vecs =[]
	ks = []
	ellBottom = np.dot(w,np.matmul(Bmat,np.matmul(K,np.matmul(Bmat,w))))
	ellLeft = np.matmul(K,np.matmul(Bmat,w))
	for i in range(len(predictX)):
		# print(i)
		t = predictX[i]
		
		k=kernelVector(dataT,t)
		ks.append(k)
		vi = np.matmul(Kinv,k)
		v.append(vi)
		phiVec = np.zeros(order+1)
		for j in range(order+1):
			phiVec[j] = phi(j,t)
		phis.append(phiVec)

		vec = np.matmul(Bmat,vi) + np.matmul(C,phiVec)
		vecs.append(vec)
		alpha[i] = np.dot(vec,dataX)
		beta[i] = np.dot(np.matmul(Bmat,w),dataX)
	
		curlyB[i,i] = beta[i]

		ellTop = np.dot(ellLeft,vec) - np.dot(k,np.matmul(Bmat,w))
		ell[i] = ellTop/ellBottom
		


	mDim = len(predictX) -1

	
	# zs = np.random.uniform(-5,-2,(mDim,1))
	# prev = np.dot(vecs[0],dataX)
	# for i in range(1,len(predictX)):
	# 	pred = np.dot(vecs[i],dataX)
	# 	naivediff = pred - prev
	# 	if naivediff > 0:
	# 		zs[i-1] = np.log(naivediff)
	# 		prev = pred
	# 	else:
	# 		zs[i-1] = -4

	
	D = np.zeros((mDim,mDim+1))
	for i in range(mDim):
		D[i,i] = -1
		D[i,i+1] = 1
	ps = alpha
	updateCs = D @ps	
	updateCs[updateCs<0] = 0
	zs = np.log(updateCs+1e-6)
	cs = np.exp(zs)


	DBDTinv = np.linalg.inv(np.matmul(D,np.matmul(curlyB,D.transpose())))
	Dalpha = np.matmul(D,alpha)
	H = np.matmul(D.transpose(),DBDTinv)
	Ht = H.transpose()
	ms = np.zeros((len(zs),1))
	vs = np.zeros((len(zs),1))
	grad = np.zeros((len(zs),1))
	for s in range(steps):
		cs = np.exp(zs)

		diff = np.matmul(H,cs-Dalpha)+ell
		
		grad = np.multiply(cs,np.matmul(Ht,diff))
		#ADAM step routine
		ms = learningMemory * ms + (1.0 - learningMemory)*grad
		vs = learningMemory_SecondMoment * vs + (1.0 - learningMemory_SecondMoment)*np.multiply(grad,grad)
		c1 = 1.0 - learningMemory**(s+1)
		c2 = 1.0 - learningMemory_SecondMoment**(s+1)
		eps = 1e-8
		zs -= learningRate * np.divide(ms/c1,np.sqrt(eps + vs/c2))
		#prevents the prediction from 'dying' by going too negative
		for j in range(1,len(zs)):
			m = -10
			if zs[j] < m:
				zs[j] = m
			l = 20
			if zs[j] > l:
				zs[j] = l
	cs= np.exp(zs)
	ps = np.zeros(len(predictX),)
	rms=0
	correct = np.matmul(H,cs-Dalpha)
	R = np.matmul(Bmat,w)
	mse = 0
	for i in range(len(predictX)):

		ai = vecs[i] + correct[i] * R
		mse += np.dot(ai,K@ai) - 2 * np.dot(ai,ks[i])
		ps[i] = np.dot(ai,dataX)
		rms += (trueY[i] - ps[i])**2
	rms = np.sqrt(rms/len(ps))
	return [ps,rms,mse]


def CLUP_Iter(predictX,dataT,dataX,order,steps):
	trueY = Func(predictX)
	K=kernelMatrix(dataT) +  dataNoise*dataNoise*np.identity(len(dataT)) #softening *kernel(0,0)* np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	w = np.matmul(Kinv,dataX)
	q = np.zeros((len(predictX),1))
	n = len(dataX)
	Phi = np.zeros((order+1,len(dataT)))
	for m in range(0,order+1):
		for j in range(len(dataT)):
			Phi[m,j] = phi(m,dataT[j])
	PhiT = Phi.transpose()
	phis = []
	v = []
	M = np.matmul(Phi,np.matmul(Kinv,PhiT))
	Minv = np.linalg.inv(M)
	C = np.matmul(np.matmul(Kinv,PhiT),Minv)
	Bmat = np.identity(n)-np.matmul(C,Phi)
	alpha = np.zeros((len(predictX),1))
	beta = np.zeros((len(predictX),1))
	curlyB = np.zeros((len(beta),len(beta)))
	ell = np.zeros((len(predictX),1))
	vecs =[]
	ks = []
	ellBottom = np.dot(w,np.matmul(Bmat,np.matmul(K,np.matmul(Bmat,w))))
	ellLeft = np.matmul(K,np.matmul(Bmat,w))
	for i in range(len(predictX)):
		# print(i)
		t = predictX[i]
		
		k=kernelVector(dataT,t)
		ks.append(k)
		vi = np.matmul(Kinv,k)
		v.append(vi)
		phiVec = np.zeros(order+1)
		for j in range(order+1):
			phiVec[j] = phi(j,t)
		phis.append(phiVec)

		vec = np.matmul(Bmat,vi) + np.matmul(C,phiVec)
		vecs.append(vec)
		alpha[i] = np.dot(vec,dataX)
		beta[i] = np.dot(np.matmul(Bmat,w),dataX)
	
		curlyB[i,i] = beta[i]

		ellTop = np.dot(ellLeft,vec) - np.dot(k,np.matmul(Bmat,w))
		ell[i] = ellTop/ellBottom

	
	mDim = len(predictX) -1
	D = np.zeros((mDim,mDim+1))
	for i in range(mDim):
		D[i,i] = -1
		D[i,i+1] = 1

	BD = curlyB @ D.transpose()
	DBDTinv = np.linalg.inv(np.matmul(D,np.matmul(curlyB,D.transpose())))
	Dalpha = np.matmul(D,alpha)
	Dt = D.transpose()
	H = np.matmul(D.transpose(),DBDTinv)
	Ht = H.transpose()
	
	ps = alpha
	print(ps)
	cs = np.zeros((mDim,1))
	# lambdaVec = np.matmul(DBDTinv, (cs - Dalpha))
	# print(lambdaVec)
	for i in range(steps):

		
		updateCs = D @ps
		
		updateCs[updateCs<0] = 0
		# aPrime = alpha + curlyB @ Dt @ (cs-updateCs)
		lambdaVec = DBDTinv @ (updateCs - D @ alpha)
		ps = alpha + BD @ lambdaVec
		cs = updateCs
	rms=0
	correct = np.matmul(H,cs-Dalpha)
	R = np.matmul(Bmat,w)
	mse = 0
	for i in range(len(predictX)):

		ai = vecs[i] + correct[i] * R
		mse += np.dot(ai,K@ai) - 2 * np.dot(ai,ks[i])


		# ps[i] = np.dot(ai,dataX)
		rms += (trueY[i] - ps[i])**2
	rms = np.sqrt(rms/len(ps))
	return [ps,rms,mse]

mode = 0

if mode == 0:
	def Transform(z):
		# return z
		out = np.zeros(np.shape(z))
		out[0] = z[0]
		for i in range(1,len(z)):
			out[i]=out[i-1] + np.exp(z[i])
		return out

	def TransformDerivative(z,i):
		val = 1
		if i > 0:
			val =np.exp(z[i])

		out = np.zeros(np.shape(z))
		out[i:] = val
		return out
	def Func(t):
		return 70.0/(1 + np.exp(-t)) +100 + 20.0/(1 + np.exp(-(t-5)*2)) 
		# return t + np.sin(4*t)
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
	
		return g
		
	def Func(t):
		sig = 1.5
		return 1.0/(np.sqrt(2*np.pi)*sig) * np.exp(- (t)**2/(2*sig**2))
xMin = -10
xMax = 10
m = (Func(xMax) - Func(xMin))/(xMax - xMin)
c = Func(xMin) -xMin *m

def Prior(t):
	return m * t + c
	# return Func(t)
	# return t-t
def PriorModifiedMSE(ell,ts,MSE):
	ts =np.sort(ts)
	meanDiff = np.mean(np.diff(ts))
	span = np.ptp(ts)

	ellMod = (ell - meanDiff)/span
	return MSE + 0.5 * ellMod**2

def GenerateData(nData):
	scatter = 0.3
	t = np.linspace(xMin,xMax,nData) + scatter * np.random.normal(0,3,nData,)
	# t = np.random.uniform(xMin,xMax,nData)
	t = np.sort(t)
	x = Func(t) + np.random.normal(0,dataNoise,nData,)
	return [t,x]

def specialShow():
	pt.draw()
	pt.pause(0.01)
	input("Enter to exit")

def optim():

	ndat = 31
	[t,x] = GenerateData(ndat)
	# c = np.mean(x)
	tt = np.linspace(min(t),max(t),100)
	res = 150
	# sigmas = np.logspace(-,1,res,10)
	sigmas = np.linspace(0.1,10,res)
	mse = np.zeros(np.shape(sigmas))
	rms = np.zeros(np.shape(sigmas))
	fig,axs = pt.subplots(3,1)
	axs[0].plot(tt,Func(tt),"k:",label="True Function")
	axs[0].scatter(t,x,label="Data")

	for i in tqdm(range(res)):
		global kernelSigma
		kernelSigma = sigmas[i]
		[ps,rmsi,msei] = C_BLP(tt,t,x,500)
		mse[i] = msei
		rms[i] = rmsi
		if i % int(res/6) == 0:
			axs[0].plot(tt,ps,label="$\\theta=$"+strRound(kernelSigma))
	mse -= np.min(mse)-1e-2
	mod = PriorModifiedMSE(sigmas,t,mse)
	prior = PriorModifiedMSE(sigmas,t,mse-mse)
	# mod -= np.min(mod)-1e-2
	prior -= np.min(prior)-1e-2

	y = np.argmin(mod)
	print("Minimum value = ", sigmas[y], "with prior centred at ",np.mean(np.diff(t)))

	axs[0].set_xlabel("t")
	axs[0].set_ylabel("X")

	axs[1].plot(sigmas,mse,label="Raw Likelihood")
	axs[1].plot(sigmas,mod,label="With Prior")
	axs[1].plot(sigmas,prior,":",label="Prior Only")
	axs[1].legend()
	axs[1].set_xscale('log')
	axs[1].set_yscale('log')
	axs[1].set_xlabel("$\\theta$")
	axs[1].set_ylabel("Global MSE")

	axs[2].plot(sigmas,rms)
	axs[2].set_xscale('log')
	axs[2].set_yscale('log')
	axs[2].set_xlabel("$\\theta$")
	axs[2].set_ylabel("True RMS")

	axs[0].legend()
	specialShow()

def iterTest():
	ndat = 10
	global kernelSigma
	kernelSigma = 2.5
	[t,x] = GenerateData(ndat)
	# c = np.mean(x)
	tt = np.linspace(min(t),max(t),100)
	res = 150
	pt.plot(tt,Func(tt),"k:",label="True Function")	
	pt.scatter(t,x,label="Data")

	order = 3
	
	[clup,rms] = BLUP(tt,t,x,order)
	s,=pt.plot(tt,clup,'--',label=str(order)+"-BLUP")
	
	for iter in [10,100,1000]:
		[clup,rms,mse] = CLUP(tt,t,x,order,iter)
		s,=pt.plot(tt,clup,'--',label=str(order)+"-CLUP-" + str(iter) +", $\mu=$" + strRound(mse))

		[clup,rms,mse] = CLUP_Boosted(tt,t,x,order,iter)
		pt.plot(tt,clup,color=s.get_color(),label=str(order)+"-CLUP_Boost-" + str(iter) +", $\mu=$" + strRound(mse))

	pt.legend()
	specialShow()
def blupTest():
	ndat = 10
	global kernelSigma
	kernelSigma = 0.1
	[t,x] = GenerateData(ndat)
	# c = np.mean(x)
	tt = np.linspace(min(t),max(t),100)
	res = 150
	pt.plot(tt,Func(tt),"k:",label="True Function")	
	pt.scatter(t,x,label="Data")
	
	
	

	global m,c
	oldm = m
	oldc = c
	m=0
	c=0
	[blp,rms] = BLP(tt,t,x)
	s, =pt.plot(tt,blp,'--',label="BLP, $\epsilon=$" + strRound(rms))
	[clp,rms,mse] = C_BLP(tt,t,x,3000)
	pt.plot(tt,clp,color=s.get_color(),label="CLP, $\epsilon=$" + strRound(rms))

	# c = np.mean(x)
	# [blp,rms] = BLP(tt,t,x)
	# s, =pt.plot(tt,blp,"--",label="BLP_Mean, $\epsilon=$" + strRound(rms))
	# [clp,rms,mse] = C_BLP(tt,t,x,3000)
	# # pt.plot(tt,Prior(tt),"r:",label="Prior")

	# pt.plot(tt,clp,color=s.get_color(),label="CLP_Mean, $\epsilon=$" + strRound(rms))

	# m=oldm
	# c = oldc
	# priort = t[1:-1]
	# priorx = x[1:-1]
	# [blp,rms] = BLP(tt,priort,priorx)
	# s, =pt.plot(tt,blp,"--",label="BLP_LinearPrior, $\epsilon=$" + strRound(rms))
	# [clp,rms,mse] = C_BLP(tt,priort,priorx,3000)
	# # pt.plot(tt,Prior(tt),"r:",label="Prior")

	# pt.plot(tt,clp,color=s.get_color(),label="CLP_LinearPrior, $\epsilon=$" + strRound(rms))

	# # for order in range(0,1):
	# # 	[blup,rms] = BLUP(tt,t,x,order)
	# # 	pt.plot(tt,blup,label=str(order)+"-BLUP, $\epsilon=$" + strRound(rms))

	# cols =[u'b', u'g', u'r', u'c', u'm', u'y', u'k']
	# i = 0
	# for order in [1,3]:
	# 	[blup,rms] = BLUP(tt,t,x,order)
	# 	s, = pt.plot(tt,blup,'--',label=str(order)+"-BLUP, $\epsilon=$" + strRound(rms))
	# 	[clup,rms,a] = CLUP(tt,t,x,order,1000)
	# 	pt.plot(tt,clup,color=s.get_color(),label=str(order)+"-CLUP, $\epsilon=$" + strRound(rms))
	# 	# [clup,rms] = CLUP(tt,t,x,order,100)
	# 	# pt.plot(tt,clup,label=str(order)+"-CLUP-low, $\epsilon=$" + strRound(rms))
	# 	i+=1
	pt.xlabel("$t$")
	pt.ylabel("$X_t$")
	pt.legend()
	specialShow()


np.random.seed(1)
blupTest()
# iterTest()
# optim()