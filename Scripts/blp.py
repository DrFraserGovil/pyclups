import numpy as np
import imageio
from matplotlib import pyplot as pt
large_width = 400
np.set_printoptions(linewidth=large_width)
np.random.seed(0) #enable for reproducable randomness

nData = 18
nPredict = 100
dataNoise = 1e-1
kernelSigma = 2e1
softening = 1e-5
learningRate = 0.1
learningMemory = 0.9
learningMemory_SecondMoment = 0.999
useFlatStart = True
maxOptimSteps = 1000
stepDelta = 1
def sig(x): #standard logit function
	return 1.0/(1.0 + np.exp(-x))

def targetFunction(x):
	#the true underlying function the data is drawn from
	return sig(x) + sig(x-7) + sig(x/2+8)

def kernel(x,y):
	#covariance the kernel
	d = abs(x-y)
	return 10.0/(1+(d/kernelSigma)**2)

def generateSample(min,max,n):
	#generates a random set {(t,X_t)}

	scale = (max-min)/n
	xs = np.linspace(min,max,n) + np.random.normal(0,scale,n) #scatter the points uniformly, then add some noise -- prevents too much grouping
	z = targetFunction(xs) 
	noise = np.random.normal(0,dataNoise,n) #add some noise to the function, just for fun
	return [xs,z+noise]

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


def basicMethod(predictX,x,y):
	#standard BLP mechanism

	#precompute some useful quantities
	K=kernelMatrix(x) + softening * np.identity(len(x)) #add softening to the diagonal for stable inversion
	Kinv = np.linalg.inv(K)
	KinvX = np.matmul(Kinv,y)

	#loop over the prediction points, and compute the prediction at each one
	ps = np.zeros(len(predictX))
	for i in range(len(predictX)):
		t = predictX[i]
		
		k=kernelVector(x,t)
		ps[i] = np.dot(KinvX,k) #normal BLP
	return ps

def piecewiseMethod(predictX,x,y):
	#similar to the the basicMethod, but goes constant whenever monotonicity is broken

	#precompute values
	K=kernelMatrix(x) + softening * np.identity(len(x)) #add softening to the diagonal for stable inversion
	Kinv = np.linalg.inv(K)
	KinvX = np.matmul(Kinv,y)
	ps = np.zeros(len(predictX))

	#loop over the prediction points, and compute the prediction at each one
	for i in range(len(predictX)):
		t = predictX[i]
		
		k=kernelVector(x,t)
		pred = np.dot(KinvX,k)
		
		#if prediction is monotonically increasing, then use it
		if i == 0 or pred > ps[i-1]:
			ps[i] = pred
		else: #otherwise, use the previous value
			ps[i] = ps[i-1]

	return ps

def trueError(xPredict,yPredict):
	#computes the RMS error between the true function and the predicted function

	trueVals = targetFunction(xPredict)
	diff = (trueVals - yPredict)
	q = np.sqrt(np.dot(diff,diff)/len(xPredict))

	#return it as a string, rounded to 3 sig fig for neatness
	return '%s' % float('%.3g' % q)

def globalFit(predictX,x,y,steps,zs=False,ms=False,vs=False):
	#execute the global fitting routine

	#some slightly hacky stuff allowing the method to resume (needed for the sequential plotting/animation)
	if type(zs) is bool:
		#if no initial state provided, generate a default one
		zs = np.zeros(len(predictX))
		zs[1:]=-10
		ms = np.zeros(len(zs))
		vs = np.zeros(len(zs))

	#precompute values as before
	K=kernelMatrix(x) + softening * np.identity(len(x))
	Kinv = np.linalg.inv(K)
	w = np.matmul(Kinv,y)

	#now precompute some values which vary with time, so store them in vectors
	kdotw = np.zeros(len(predictX))
	As = np.zeros(len(predictX))
	ks = [np.zeros(len(x))]*len(predictX)
	for i in range(len(predictX)):
		t = predictX[i]
		
		k=kernelVector(x,t)
		ks[i] = k
		v = np.matmul(Kinv,k)
		As[i] = np.dot(v,y)
		kdotw[i] = np.dot(k,w)


	#iterate over a number of steps, computing the optimisation step per the ADAM optimiser
	#learning rate and memory parameters defined globally for ease
	for s in range(steps):

		#generate empty gradient vector
		grad = np.zeros(len(zs))

		#convert zs into prediction values
		deltas = np.append(zs[0],np.exp(zs[1:])) 
		S = np.cumsum(deltas)

		#this is my attempt to efficiently compute the gradient calculations in Eq. 31
		bracket = (2*S - As - kdotw)
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
	
	#returns both the prediction (zs), and the internal parameters used for resuming the optimiser
	qs = zs.copy()
	for j in range(1,len(zs)):
		qs[j] = zs[j]
		zs[j] = zs[j-1] + np.exp(zs[j])
	return [zs,qs,ms,vs]


#generate and plot the raw data
[xx,yy] = generateSample(-20,30,nData)
mu = np.mean(yy) #shift the data by the mean - remember to add it back in!
yShift= yy - mu
pt.scatter(xx,yy)

#generate the prediction points -- I make them stick over the edge of the data just to see how bad the extrapolation is
r = max(xx) - min(xx)
f = 0.01
predictX = np.linspace(min(xx) - f*r,max(xx)+f*r,nPredict)

#generate and plot the True Function
xR = np.linspace(min(xx),max(xx),1000)
yR = targetFunction(xR)
pt.plot(xR,yR,':',label="True Value")


#generate then plot the basic BLP method
pNaive = basicMethod(predictX,xx,yShift) + mu #reshifted back to the mean!
pt.plot(predictX,pNaive,linewidth=3,label="Naive solution, err=" +trueError(predictX,pNaive))
pt.draw()
pt.legend()
pt.pause(0.01)

#generate then plot the stepwise-monotonic method
pPiecwise = piecewiseMethod(predictX,xx,yShift) + mu
pt.plot(predictX,pPiecwise,label="Piecewise solution,err="+trueError(predictX,pPiecwise))
pt.draw()
pt.legend()
pt.pause(0.01)


##this looks daunting, but it's just complex because I run it through a sequential loop so it can be plotted into a nice animation

#we can chose to use either a naive (flat) initialisation, or we can use the piecewise solution as a starting point. 
if useFlatStart == False:
	initGuess = np.zeros(np.shape(pPiecwise))
	rms = np.zeros(np.shape(pPiecwise))
	rvs = np.zeros(np.shape(pPiecwise))

	#convert piecewise solution into z coordinates
	initGuess[0] = pPiecwise[0] - mu
	for i in range(1,len(initGuess)):
		initGuess[i] = np.log(1e-9+ pPiecwise[i] - pPiecwise[i-1])

#needed to keep the y bounds nice during the loop
curMin = min(min(yy),min(pPiecwise),min(pNaive))
curMax = max(max(yy),max(pPiecwise),max(pNaive))
i = 0
times = []
# loop over the total number of steps
for s in range(0,maxOptimSteps,stepDelta):

	if i > 0:
		#runs over (s-prev) steps, to get the total number of optimisation steps to s
		[pGlobal,rzs,rms,rvs] = globalFit(predictX,xx,yShift,s-prev,rzs,rms,rvs) 
		pGlobal +=mu
		ln.remove() #removes the previous line from the canvas, allows animation
	else:
		[pGlobal,rzs,rms,rvs] = globalFit(predictX,xx,yShift,s) #run using default initialisatoin
		# [pGlobal,rzs,rms,rvs] = globalFit(predictX,xx,yShift,s,initGuess,rms,rvs) #run using the piecewise result as the init
		pGlobal += mu
	i+=1
	prev = s

	#various plotting nonsense
	ln, = pt.plot(predictX,pGlobal,label="Global solution " +str(s)+ ", err=" +trueError(predictX,pGlobal),color="red")
	pt.draw()
	pt.ylim(min(min(pGlobal),curMin)-0.2,max(max(pGlobal),curMax)+0.2)
	pt.legend()
	pt.pause(0.001)

# 	#save to file, allowing gif generation
# 	pt.savefig(f'tmp/frame_{i}')
# 	times.append(i)

# #the gif generation loop
# frames = []
# for t in times:
# 	image = imageio.v2.imread(f'tmp/frame_{t}.png')
# 	frames.append(image)
# imageio.mimsave('./example.gif',frames, duration=20)  

#stops the code exiting and shutting the window down
input("Return to exit")