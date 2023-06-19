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
kernelSigma = 1e1
softening = 1e-3
learningRate = 0.1
learningMemory = 0.9
learningMemory_SecondMoment = 0.999
useFlatStart = True
maxOptimSteps = 1000
stepDelta = 1

genMean = 0
genSigma = 0.2

printMode = True
printMode = False
funcScale = 0


# np.random.seed(0)
# printMode = True
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



def generateArbitraryFunc(xs):
	r = np.random.randint(0,2)
	genSigma = np.exp(np.log(10) * np.random.uniform(0,3))
	if r == 0:
		return genMean + genSigma *np.random.standard_cauchy(len(xs))
	else:
		return genMean + 5*genSigma * np.random.standard_normal(len(xs))
def sampleArbitraryFunc(ws):
	global funcScale
	funcScale= np.power(10,np.random.uniform(-0.5,2))
	ys = np.zeros(len(ws))
	ys[0] = np.random.uniform(-5,5)
	for i in range(1,len(ws)):
		ys[i] =  ws[i] +ys[i-1]
	ys -= np.min(ys)
	
	zs = np.cumsum(ys)
	return zs *funcScale/zs[-1]

def naiveScore(predictX,dataT,dataX,trueY):
	#standard BLP mechanism
	mu = np.mean(dataX)
	
	#precompute some useful quantities
	K=kernelMatrix(dataT) + (dataNoise * dataNoise) * np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	KinvX = np.matmul(Kinv,dataX-mu)

	#loop over the prediction points, and compute the prediction at each one
	ps = np.zeros(len(predictX))
	rms = 0
	for i in range(len(predictX)):
		t = predictX[i]
		
		k=kernelVector(dataT,t)
		ps[i] =mu +  np.dot(KinvX,k) #normal BLP
		rms += (trueY[i] - ps[i])**2

	if printMode:
		pt.plot(predictX,ps,label="Basic")
	return np.sqrt(rms/len(predictX))
def stupidScore(predictX,dataT,dataX,trueY):
	#standard BLP mechanism
	mu = np.mean(dataX)
	
	#precompute some useful quantities
	K=kernelMatrix(dataT) + softening * kernel(0,0) * np.identity(len(dataT)) #add softening to the diagonal for stable inversion
	Kinv = np.linalg.inv(K)
	KinvX = np.matmul(Kinv,dataX-mu)

	#loop over the prediction points, and compute the prediction at each one
	ps = np.zeros(len(predictX))
	rms = 0
	for i in range(len(predictX)):
		t = predictX[i]
		
		k=kernelVector(dataT,t)
		test = mu + np.dot(KinvX,k) #normal BLP

		if i == 0 or test > ps[i-1]:
			ps[i] = test
		else:
			ps[i] = ps[i-1]

		rms += (trueY[i] - ps[i])**2
		# print("  ",i,rms,trueY[i],ps[i])
	# pt.plot(predictX,ps,label="Stupid")

	rms = np.sqrt(rms/len(predictX))
	return rms

def globalScore(predictX,dataT,dataX,trueY,steps):
	zs = np.zeros(len(predictX))
	zs[1:]=-10
	ms = np.zeros(len(zs))
	vs = np.zeros(len(zs))

	#precompute values as before
	K=kernelMatrix(dataT) + (dataNoise * dataNoise) * np.identity(len(dataT)) #softening *kernel(0,0)* np.identity(len(dataT))
	Kinv = np.linalg.inv(K)
	mu = np.mean(dataX)
	w = np.matmul(Kinv,dataX-mu)

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
		As[i] = np.dot(v,dataX-mu)
		kdotw[i] = np.dot(k,w)

		test = np.dot(w,k)
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
		bracket = 2*(S - As)
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
			m = -50
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
	if printMode:
		pt.plot(predictX,ps,label="Global " + str(steps))
	return np.sqrt(rms/len(predictX))





def meanScore(kSigma, dataSigma, dataCount,N):
	nData = dataCount
	xs = np.linspace(0,100,200)
	loops = range(0,N)
	idx = range(0,len(xs))
	naive = np.zeros(N)
	clever = np.zeros(N)
	global kernelSigma, dataNoise
	kernelSigma = kSigma
	
	for i in loops:
		# print(i)
		
		ws = generateArbitraryFunc(xs)
		ys =sampleArbitraryFunc(ws)
		dataNoise = dataSigma * funcScale
		sampleIdx = np.sort(np.random.choice(idx,nData,replace=False))
		subx = xs[sampleIdx[0]:sampleIdx[-1]] #only perform predictions within the sampled data -- do not extrapolate
		# suby = ys[sampleIdx[0]:sampleIdx[-1]]
		noise = np.random.normal(0,dataNoise,nData)
		dataT = xs[sampleIdx]
		dataX = ys[sampleIdx] + noise

		if printMode:
			pt.figure(2)
			pt.clf()
			pt.plot(xs,ys,label="True function")
			pt.scatter(xs[sampleIdx],ys[sampleIdx]+noise,label="Sample")

		naive[i] = naiveScore(subx,dataT,dataX,ys)
		clever[i] = globalScore(subx,dataT,dataX,ys,1000)
	
		if printMode:
			pt.legend()
			pt.draw()
			pt.pause(0.01)
	nMean = np.median(naive)
	nClever= np.median(clever)

	ratio = np.divide(naive,clever)
	nAmp = np.mean(ratio) -1
	noImprove= ratio < 0.999
	if np.count_nonzero(noImprove) == 0:
		failure = np.nan
	else:
		failure = np.mean(ratio[noImprove])-1
	return [nAmp, failure]

multiPlot = False
if multiPlot:

	N = 150
	# nData = 15
	grid = 15
	kSigs = np.linspace(1,20,grid)
	dSigs = np.power(10,np.linspace(-3,-0.5,grid))

	yy,xx = np.meshgrid(kSigs,dSigs)
	pt.ion()
	fig,axs = pt.subplots(2,1)
	fig.set_size_inches(8.5, 8.5)
	colExist = False
	for nData in tqdm(range(5,50,5)):
		axs[0].cla()
		axs[1].cla()
		amplification = np.zeros((grid,grid)) + np.nan
		failure = np.zeros((grid,grid)) + np.nan
		
		
		for i in tqdm(range(grid),leave=False):
			for j in tqdm(range(grid),leave=False):
				[a,b] = meanScore(kSigs[i],dSigs[j],nData,N)
				amplification[j][i] = max(a,1e-3)
				failure[j][i] = abs(b)
				# print(kSigs[i],dSigs[j],c)



				# if printMode:
				# fig = pt.figure(1)
				
				# pt.subplot(0)
				top = np.nanmax(amplification)
				bottom = np.nanmin(amplification)
				c = axs[0].pcolormesh(xx,yy,amplification,norm=colors.LogNorm(vmin=bottom,vmax=top))
				axs[0].set_xscale('log')
				axs[0].set_title("Amplification with " + str(nData) + " samples")
				# pt.yscale('log')
				# axs[0].set_xlabel("Data Variation")
				axs[0].set_ylabel("Kernel Smoothing Length")
				# axs[0].set_axis([xx.min(), xx.max(), yy.min(), yy.max()])

				# if np.any(not np.isnan(failure))
				top = np.nanmax(failure)
				bottom = np.nanmin(failure)
				if np.isnan(top) or np.isnan(bottom):
					bottom = 0.01
					top = 0.02
				# print(top,bottom)
				f = axs[1].pcolormesh(xx,yy,failure,norm=colors.LogNorm(vmin=bottom,vmax=top))
				axs[1].set_xscale('log')
				axs[1].set_title("BLMP Failure Analysis")
				# pt.yscale('log')
				axs[1].set_xlabel("Data Variation")
				axs[1].set_ylabel("Kernel Smoothing Length")
				# axs[1].set_axis([xx.min(), xx.max(), yy.min(), yy.max()])

				if colExist:
					q.update_normal(c)
					q2.update_normal(f)
				else:
					q = pt.colorbar(c,label="RMS-amplification relative to BLP")
					q2 = pt.colorbar(f,label="RMS-suppression relative to BLP")
					colExist = True
				pt.draw()
				pt.pause(0.02)
		pt.savefig("test"+str(nData)+".jpg")
	# pt.clf()
else:
	printMode = True
	nData = 20
	[a,b] = meanScore(15,0.1,nData,1)

input("Enter to exit")