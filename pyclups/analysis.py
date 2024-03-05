import pyclups
import numpy as np
from matplotlib import colors as mpcol
from matplotlib import pyplot as pt
from tqdm import tqdm as tqdm

def curvy(x,params):
	# v = np.sum([params[i]*np.power(x,i) for i in range(len(params))],0)
	# l = np.cumsum(np.maximum(0,v))

	l = 1/(1 + np.exp(-x))

	return l

def RunTest(**kwargs):
	pt.rcParams['text.usetex'] = True
	nData = lambda : np.random.randint(7,20)
	curveGenerator = lambda x,params: curvy(x,params)  #monotonic function
	noise = 0.3
	basis = pyclups.basis.Hermite(5)
	constraint = pyclups.constraint.Monotonic()
	# constraint.Add(pyclups.constraint.Positive(lambda t: np.abs(t +10) < 1e-3))
	outRes = 25
	K_generator = lambda ell: pyclups.kernel.SquaredExponential(kernel_variance=0.5,kernel_scale=ell)
	saveName = "output"
	description = "Monotonic EIE"
	for key,value in kwargs.items():
		if key=="data_count":
			nData = value
		elif key == "curve":
			curveGenerator = value
		elif key == "noise":
			noise = value
		elif key == "basis":
			basis = value
		elif key == "constraint":
			constraint = value
		elif key == "resolution":
			outRes = value
		elif basis=="resolution":
			K_generator = value
		else:
			raise KeyError("Unknown key (" + str(key) + ") passed to Analysis Interface")
		
	ellMin = 0.01
	ellMax = 2
	ellRes = 25
	noiseMin = 0.01
	noiseMax = 0.5
	noiseRes = 25
	fails = 0
	success = 0
	sampling = 25

	ells = np.linspace(ellMin,ellMax,ellRes)
	noises=np.linspace(noiseMin,noiseMax,noiseRes)
	successGrid = np.zeros((noiseRes,ellRes))
	failAmountGrid = np.zeros((noiseRes,ellRes))
	succeedAmountGrid =np.zeros((noiseRes,ellRes))
	counts = np.zeros((noiseRes,ellRes))
	count = 0

	made=False
	quiet = False
	figs,axs = pt.subplots(3,1,figsize=(8,12))
	
	
	count = 0
	allCount = 0
	coincidentCount = 0
	for k in tqdm(range(sampling),leave=False,disable=quiet):
		if callable(nData):
			N = nData()
		else:
			N = nData
		params = 3 *np.random.random((2,))
		x = np.linspace(-10,10,N)
		ttOrig = np.linspace(x[0],x[-1],outRes)
		
		
		for j in tqdm(range(0,ellRes),leave=False,disable=quiet):
			ell = ells[j]
				
			K = K_generator(ell)	
			s = pyclups.Predictor(K,constraint,basis)
			
			for i in tqdm(range(0,noiseRes),leave=False,disable=quiet):
				noise = noises[i]
				xnoise = np.random.normal(0,noise,(outRes,))
				tt = np.sort(ttOrig+xnoise)
				true = curveGenerator(tt,params)
				yTrue = curveGenerator(x,params)

				y = yTrue + np.random.normal(0,noise,(N,))
				pred = s.Predict(tt,x,y,noise)
				
				allCount += 1
				e_clups = 0
				e_blups = 0
				for q in range(outRes):
					e_clups += (true[q] - pred.X[q])**2
					e_blups += (true[q] - pred.X_BLUP[q])**2
				e_clups=np.sqrt(e_clups)/outRes
				e_blups=np.sqrt(e_blups)/outRes
				improvement = (e_blups-e_clups)/e_clups
				
				if np.abs(improvement) > 1e-4:
					counts[i,j] += 1
					count += 1
					
					if improvement < 0:
						failAmountGrid[i,j] += np.abs(improvement)
					else:
						successGrid[i,j] +=1
						succeedAmountGrid[i,j] += np.abs(improvement)
				else:
					coincidentCount+=  1
	if made == True:
		cb1.remove()
		cb2.remove()
		cb3.remove()
	axs[0].cla()
	axs[1].cla()
	axs[2].cla()
	r =  f"({100*(allCount - coincidentCount)/allCount:.0f}\% non-coincident CLUPS)"
	tit = description + " " + r
	figs.suptitle(tit,fontsize=15)
	im=axs[0].imshow(successGrid/counts,extent = [ells[0],ells[-1],noises[0],noises[-1]], aspect='auto')
	cb1 = pt.colorbar(im,ax=axs[0])
	cb1.set_label("Fraction of CLUPS Equal or Better than BLUPS")
	im=axs[1].imshow(succeedAmountGrid/successGrid,extent = [ells[0],ells[-1],noises[0],noises[-1]],norm=mpcol.LogNorm(), aspect='auto')
	cb2 = pt.colorbar(im,ax=axs[1])
	cb2.set_label("Mean CLUPS Improvement Amount")
	if np.any(failAmountGrid):
		im=axs[2].imshow(failAmountGrid/(counts - successGrid),extent = [ells[0],ells[-1],noises[0],noises[-1]],norm=mpcol.LogNorm(), aspect='auto')
		cb3 = pt.colorbar(im,ax=axs[2])
	else:
		im=axs[2].imshow(failAmountGrid/(count - successGrid),extent = [ells[0],ells[-1],noises[0],noises[-1]],norm=mpcol.LogNorm(1e-10,1e-5), aspect='auto')
		cb3 = pt.colorbar(im,ax=axs[2])
	made = True
	cb3.set_label("Mean CLUPS Failure Amount")
	axs[2].set_xlabel("Kernel Length Scale")
	axs[0].set_ylabel("Gaussian Noise, $\sigma_E$")
	axs[1].set_ylabel("Gaussian Noise, $\sigma_E$")
	axs[2].set_ylabel("Gaussian Noise, $\sigma_E$")
	figs.tight_layout()
	# pt.draw()
	# pt.pause(0.01)
	pt.savefig(f"{saveName}.pdf",format='pdf')