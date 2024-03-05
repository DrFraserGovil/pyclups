import numpy as np

def GenerateData(**kwargs):
	#synthesises a sample from Func()

	# scatter = 0.9
	N = 10
	func = lambda x: 1.0/(1 + np.exp(-x))
	xmin = 0
	xmax = 10
	dataNoise = 0.2
	mode = "uniform"
	for key,value in kwargs.items():
			if key == "n":
				N = value
			elif key == "function":
				func = value
			elif key == "xmin":
				xmin = value
			elif key == "xmax":
				xmax = value
			elif key == "noise":
				dataNoise = value
			elif key == "mode":
				mode = value
			else:
				raise KeyError("Unknown key (" + str(key) + ") passed to kernel")

	#one of three ways to generate the x-axis points; uniform is a nice uniform grid. Random is a uniform random selection. Semi is halfway between - random noise added to the uniform grid, ensures no big blank areas.
	if mode == "uniform":
		t = np.linspace(xmin,xmax,N)
	elif mode =="random":
		t = np.random.uniform(xmin,xmax,(N,))
	elif mode == "semi":
		t = np.linspace(xmin,xmax,N) + (xmin-xmax)/(1.5*N)*np.random.normal(0,1,N,)
	else:
		raise KeyError("Unknown mode (" + mode +") passed to data generator")
	t = np.sort(t)
	x = func(t) + np.random.normal(0,dataNoise,N,)
	return [t,x]