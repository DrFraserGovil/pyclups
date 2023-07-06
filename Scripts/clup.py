#!/usr/bin/python3
import numpy as np
from matplotlib import pyplot as pt
from tqdm import tqdm
import warnings
from scipy import special
large_width = 400
np.set_printoptions(linewidth=large_width)
warnings.filterwarnings("ignore")
kernelSigma = 0.35

dataNoise = 0.7
kernelNoise = dataNoise*10
learningRate = 0.05
learningMemory = 0.7
learningMemory_SecondMoment = 0.99

#standardised container for predictor output
class Predictor:
	def __init__(self,t,p,rms,mse):
		self.T = t
		self.X = p
		self.RMS = rms
		self.MSE = mse


def Prior(t):
	return m * t + c

def phi(i,t):
	#these are the basis functions used for the BLUP/CLUP
	p_monic = special.hermite(i, monic=True)
	return p_monic(t)

def specialShow():
	#makes using pyplot somewhat bearable
	pt.draw()
	pt.pause(0.01)
	input("Enter to exit")

def kernel(x,y):
	#covariance the kernel
	d = abs(x-y)/kernelSigma
	return kernelNoise**2 * (np.exp(-0.5 * d**2))

def kernelMatrix(sampleX):
	#my attempt at computing K_ij, the covariance/ second moment matrix evaluated over the data
	n = len(sampleX)
	K = np.zeros(shape=(n,n))
	for i in range(n):
		for j in range(n):
			K[i,j] = kernel(sampleX[i],sampleX[j])
		K[i,i] += dataNoise**2 
	return K

def kernelVector(sampleX,t):
	#my attempt at computing k_i, the covariance/ second moment vector evaluated over the data, at time t
	n = len(sampleX)
	k = np.zeros(shape=(n,))
	for i in range(n):
		k[i] = kernel(sampleX[i],t)
	return k

def strRound(q):
	#rounds number to 3sf, and converts to string for easy output
	return '%s' % float('%.3g' % q)

def BLP(predictT,dataT,dataX):
	#computes the standard BLP with a prior transform

	
	#precompute some useful quantities
	muData = Prior(dataT)
	K=kernelMatrix(dataT)
	Kinv = np.linalg.inv(K)

	#loop over the prediction points, and compute the prediction at each one
	mean = Prior(predictT)
	ps = np.zeros(len(predictT))
	rms = 0
	mse = 0
	trueY = Func(predictT)
	for i in range(len(predictT)):
		t = predictT[i]
		k=kernelVector(dataT,t)
		a = Kinv @ k

		ps[i] = np.dot(a,dataX-muData) + mean[i] #normal BLP

		rms += (trueY[i] - ps[i])**2
		mse += np.dot(a,K@a) - 2*np.dot(a,k)
	rms = np.sqrt(rms/len(ps))

	return Predictor(predictT,ps,rms,mse)

def BLUP(predictX,dataT,dataX,order):
	#computes the BLUP using phi as a basis, up to the polynomial of order 'order'

	#precompute some useful quantities
	K=kernelMatrix(dataT)
	Kinv = np.linalg.inv(K)
	Phi = np.zeros((order+1,len(dataT))) #basis matrix at each of the sample points
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
	mse = 0
	for i in range(len(predictX)):
		t = predictX[i]
		
		for j in range(order+1):
			phiVec[j] = phi(j,t)

		
		k = kernelVector(dataT,t)
	
		a = np.matmul(mat1,k) + np.matmul(np.matmul(Kinv,obj),phiVec)
		
		ps[i] = np.dot(a,dataX)
		rms += (trueY[i] - ps[i])**2
		mse += np.dot(a,K@a) - 2* np.dot(k,a)
	rms = np.sqrt(rms/len(ps))
	return Predictor(predictX,ps,rms,mse)

def CLP(predictX,dataT,dataX,steps):
	gPredict = Prior(predictX)
	trueY = Func(predictX)
	tData = dataX - Prior(dataT)
	K=kernelMatrix(dataT) #softening *kernel(0,0)* np.identity(len(dataT))
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
	mDim = len(predictX) -1


	D = np.zeros((mDim,mDim+1))
	for i in range(mDim):
		D[i,i] = -1
		D[i,i+1] = 1
	ps = q
	updateCs = D @ps	
	updateCs[updateCs<0] = 0
	zs = np.log(updateCs+1e-6)
	cs = np.exp(zs)

	
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
	return Predictor(predictX,bestPredict,np.sqrt(rms),mse)

def CLUP(predictX,dataT,dataX,order,steps):

	trueY = Func(predictX)
	K=kernelMatrix(dataT) #softening *kernel(0,0)* np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	w = np.matmul(Kinv,dataX)
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
	print("ell = ",ell.transpose())
	mDim = len(predictX) -1	
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
	return Predictor(predictX,ps,rms,mse)

def even_CLUP(predictX,dataT,dataX,order,steps):

	trueY = Func(predictX)
	K=kernelMatrix(dataT) #softening *kernel(0,0)* np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	w = np.matmul(Kinv,dataX)
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
	ell = np.zeros((len(predictX),1))
	vecs =[]
	ks = []
	ellBottom = np.dot(w,np.matmul(Bmat,np.matmul(K,np.matmul(Bmat,w))))
	ellLeft = np.matmul(K,np.matmul(Bmat,w))
	beta = np.dot(np.matmul(Bmat,w),dataX)
	for i in range(len(predictX)):
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
		
	

		ellTop = np.dot(ellLeft,vec) - np.dot(k,np.matmul(Bmat,w))
		ell[i] = ellTop/ellBottom
		
	mDim = int((len(predictX) -1	)/2)
	print(mDim)
	D = np.zeros((mDim,len(predictX)))
	for i in range(mDim):
		D[i,i] = -1
		offset = len(predictX)-1
		D[i,offset-i] = 1

	cs = np.zeros((mDim,1))
	ps = alpha

	DDTinv = np.linalg.inv(np.matmul(D,D.transpose()))
	Dalpha = np.matmul(D,alpha)
	H = np.matmul(D.transpose(),DDTinv)
	Ht = H.transpose()
	
	corrector = H@(cs - D@alpha)
	bkX = Bmat@Kinv@dataX
	rms=0
	mse = 0
	for i in range(len(predictX)):
		k = kernelVector(dataT,predictX[i])

		ablp = Kinv @ k
		ablup = Bmat @ ablp + C@phis[i]




		ai = ablup + corrector[i]/beta * bkX
		mse += np.dot(ai,K@ai) - 2 * np.dot(ai,ks[i])
		ps[i] = np.dot(ai,dataX)
		rms += (trueY[i] - ps[i])**2
	rms = np.sqrt(rms/len(ps))
	return Predictor(predictX,ps,rms,mse)

def doubleEven_CLUP(predictX,dataT,dataX,order,steps):

	trueY = Func(predictX)
	K=kernelMatrix(dataT) #softening *kernel(0,0)* np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	w = np.matmul(Kinv,dataX)
	n = len(dataX)
	Phi = np.zeros((order+1,len(dataT)))
	for m in range(0,order+1):
		for j in range(len(dataT)):
			Phi[m,j] = phi(2*m,dataT[j])
	PhiT = Phi.transpose()
	phis = []
	v = []
	M = np.matmul(Phi,np.matmul(Kinv,PhiT))
	Minv = np.linalg.inv(M)
	C = np.matmul(np.matmul(Kinv,PhiT),Minv)
	Bmat = np.identity(n)-np.matmul(C,Phi)
	alpha = np.zeros((len(predictX),1))
	ell = np.zeros((len(predictX),1))
	vecs =[]
	ks = []
	ellBottom = np.dot(w,np.matmul(Bmat,np.matmul(K,np.matmul(Bmat,w))))
	ellLeft = np.matmul(K,np.matmul(Bmat,w))
	beta = np.dot(np.matmul(Bmat,w),dataX)
	for i in range(len(predictX)):
		t = predictX[i]
		
		k=kernelVector(dataT,t)
		ks.append(k)
		vi = np.matmul(Kinv,k)
		v.append(vi)
		phiVec = np.zeros(order+1)
		for j in range(order+1):
			phiVec[j] = phi(2*j,t)
		phis.append(phiVec)

		vec = np.matmul(Bmat,vi) + np.matmul(C,phiVec)
		vecs.append(vec)
		alpha[i] = np.dot(vec,dataX)
		
	

		ellTop = np.dot(ellLeft,vec) - np.dot(k,np.matmul(Bmat,w))
		ell[i] = ellTop/ellBottom
		
	mDim = int((len(predictX) -1	)/2)
	print(mDim)
	D = np.zeros((mDim,len(predictX)))
	for i in range(mDim):
		D[i,i] = -1
		offset = len(predictX)-1
		D[i,offset-i] = 1

	cs = np.zeros((mDim,1))
	ps = alpha

	DDTinv = np.linalg.inv(np.matmul(D,D.transpose()))
	Dalpha = np.matmul(D,alpha)
	H = np.matmul(D.transpose(),DDTinv)
	Ht = H.transpose()
	
	corrector = H@(cs - D@alpha)
	bkX = Bmat@Kinv@dataX
	rms=0
	mse = 0
	for i in range(len(predictX)):
		k = kernelVector(dataT,predictX[i])

		ablp = Kinv @ k
		ablup = Bmat @ ablp + C@phis[i]




		ai = ablup + corrector[i]/beta * bkX
		mse += np.dot(ai,K@ai) - 2 * np.dot(ai,ks[i])
		ps[i] = np.dot(ai,dataX)
		rms += (trueY[i] - ps[i])**2
	rms = np.sqrt(rms/len(ps))
	return Predictor(predictX,ps,rms,mse)


def Func(t):
	return t*t - 0.01*t**4 + 3*np.cos(3*t)
	# return 70.0/(1 + np.exp(-t)) +100 + 40.0/(1 + np.exp(-(t-5)*2)) 
	# return t + (t+10)**2 - (t/3.142)**4

def GenerateData(nData):
	#synthesises a sample from Func()
	#two methods of choosing x points -- either clustered, or totally uniform
	scatter = 0.9
	t = np.random.uniform(xMin,xMax,(nData,))
	# t = np.linspace(xMin,xMax,nData) + scatter*np.random.normal(0,1,nData,)
	x = Func(t) + np.random.normal(0,dataNoise,nData,)
	return [t,x]

xMin = -10
xMax = 10
m = (Func(xMax) - Func(xMin))/(xMax - xMin)
c = Func(xMin) -xMin *m


def blupTest():
	ndat = 20
	global kernelSigma
	kernelSigma = 1
	[t,x] = GenerateData(ndat)
	tt = np.linspace(min(t),max(t),50)
	pt.plot(tt,Func(tt),"k:",label="True Function")	
	pt.scatter(t,x,label="Data")
	
	global m,c

	m=0
	c=0
	blp = BLP(tt,t,x)
	s, =pt.plot(blp.T,blp.X,'--',label="BLP, $\epsilon=$" + strRound(blp.RMS))
	clp = CLP(tt,t,x,1000)
	pt.plot(clp.T,clp.X,color=s.get_color(),label="CLP, $\epsilon=$" + strRound(clp.RMS))

	c = np.mean(x)
	blp = BLP(tt,t,x)
	s, =pt.plot(blp.T,blp.X,'--',label="BLP_Mean, $\epsilon=$" + strRound(blp.RMS))
	clp = CLP(tt,t,x,3000)
	pt.plot(clp.T,clp.X,color=s.get_color(),label="CLP_Mean, $\epsilon=$" + strRound(clp.RMS))

	m=(x[-1] - x[0])/(t[-1]- t[0])
	c = x[0] - m* t[0]
	priort = t[1:-1]
	priorx = x[1:-1]
	blp = BLP(tt,t,x)
	s, =pt.plot(blp.T,blp.X,'--',label="BLP_LinearPrior, $\epsilon=$" + strRound(blp.RMS))
	clp = CLP(tt,t,x,3000)
	pt.plot(clp.T,clp.X,color=s.get_color(),label="CLP_LinearPrior, $\epsilon=$" + strRound(clp.RMS))

	for order in [1,3,8]:
		blup = BLUP(tt,t,x,order)
		s, = pt.plot(blup.T,blup.X,'--',label=str(order)+"-BLUP, $\epsilon=$" + strRound(blup.RMS))
		clup = CLUP(tt,t,x,order,1000)
		pt.plot(clup.T,clup.X,color=s.get_color(),label=str(order)+"-CLUP, $\epsilon=$" + strRound(clup.RMS))

	pt.xlabel("$t$")
	pt.ylabel("$X_t$")
	pt.legend()
	specialShow()

def evenTest():
	
	# def Func(t):
	# 	return t**2

	ndat = 40
	global kernelSigma
	kernelSigma = 1
	[t,x] = GenerateData(ndat)
	extern = max(max(t),-min(t))
	print(extern)
	tt = np.linspace(-extern,extern,171)
	pt.plot(tt,Func(tt),"k:",label="True Function")	
	pt.scatter(t,x,label="Data")
	
	global m,c

	m=0
	c=0
	blp = BLP(tt,t,x)
	s, =pt.plot(blp.T,blp.X,'--',label="BLP, $\epsilon=$" + strRound(blp.RMS))
	# clp = CLP(tt,t,x,1000)
	# pt.plot(clp.T,clp.X,color=s.get_color(),label="CLP, $\epsilon=$" + strRound(clp.RMS))

	# c = np.mean(x)
	# blp = BLP(tt,t,x)
	# s, =pt.plot(blp.T,blp.X,'--',label="BLP_Mean, $\epsilon=$" + strRound(blp.RMS))
	# # clp = CLP(tt,t,x,3000)
	# # pt.plot(clp.T,clp.X,color=s.get_color(),label="CLP_Mean, $\epsilon=$" + strRound(clp.RMS))

	# m=(x[-1] - x[0])/(t[-1]- t[0])
	# c = x[0] - m* t[0]
	# priort = t[1:-1]
	# priorx = x[1:-1]
	# blp = BLP(tt,t,x)
	# s, =pt.plot(blp.T,blp.X,'--',label="BLP_LinearPrior, $\epsilon=$" + strRound(blp.RMS))
	# clp = CLP(tt,t,x,3000)
	# pt.plot(clp.T,clp.X,color=s.get_color(),label="CLP_LinearPrior, $\epsilon=$" + strRound(clp.RMS))

	for order in [1,3]:
		blup = BLUP(tt,t,x,order)
		# s, = pt.plot(blup.T,blup.X,'--',label=str(order)+"-BLUP, $\epsilon=$" + strRound(blup.RMS))
		clup = even_CLUP(tt,t,x,2*order,1000)
		s,=pt.plot(clup.T,clup.X,':',label=str(2*order)+"-e-CLUP, $\epsilon=$" + strRound(clup.RMS))
		clup = doubleEven_CLUP(tt,t,x,order,1000)
		pt.plot(clup.T,clup.X,color=s.get_color(),label=str(order)+"-doubleE-CLUP, $\epsilon=$" + strRound(clup.RMS))

	pt.xlabel("$t$")
	pt.ylabel("$X_t$")
	pt.legend()
	specialShow()

np.random.seed(2)
blupTest()
# evenTest()
