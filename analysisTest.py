#!/opt/homebrew/bin/python3
import numpy as np
from matplotlib import pyplot as pt
import pyclups
import sys
# np.random.seed(0)


def sigmoid(x,params):
	return np.exp(x)
def exper(x,params):
	return 1.0/np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)
key = 0
if len(sys.argv) > 1:
	key = int(sys.argv[1])

high = 50
low = 30
if key == 0:
	pyclups.analysis.RunTest(
		constraint=pyclups.constraint.Monotonic(),
		resolution=75,
		bounds=[-10,10],
		basis=pyclups.basis.Hermite(3),
		curve=sigmoid,
		output ="monotonic",
		description="Monotonic EIE",
		sampling_rate=low
	)
if key == 1:
	constraint = pyclups.constraint.Monotonic()
	constraint.Add(pyclups.constraint.Positive(lambda t: t == -10))
	pyclups.analysis.RunTest(
		constraint=constraint,
		resolution=75,
		bounds=[-10,10],
		basis=pyclups.basis.Hermite(3),
		curve=sigmoid,
		output ="mono_positive",
		description="Monotonic \& Positive EIE",
		sampling_rate=low
	)
if key == 2:
	pyclups.analysis.RunTest(
		constraint=pyclups.constraint.Even(),
		resolution=50,
		bounds=[-3,3],
		basis=pyclups.basis.Hermite(3),
		curve=exper,
		output ="even",
		description="Even EIE",
		xnoise=0,
		sampling_rate = high
	)
if key == 3:
	constraint = pyclups.constraint.Even()
	constraint.Add(pyclups.constraint.Positive(lambda t: t <= 0))
	pyclups.analysis.RunTest(
		constraint=constraint,
		resolution=75,
		bounds=[-3,3],
		basis=pyclups.basis.Hermite(3),
		curve=exper,
		output ="even_positive",
		description="Even \& Positive EIE",
		xnoise=0,
		sampling_rate = low
	)
if key == 4:
	pyclups.analysis.RunTest(
		constraint=pyclups.constraint.Integrable(1.0),
		resolution=75,
		bounds=[-3,3],
		basis=pyclups.basis.Hermite(3),
		description="Integrable EIE",
		curve=exper,
		output ="integrable",
		xnoise=0,
		sampling_rate = high
	)
if key == 5:
	constraint = pyclups.constraint.Integrable(1.0)
	constraint.Add(pyclups.constraint.Even())
	pyclups.analysis.RunTest(
		constraint=constraint,
		resolution=75,
		bounds=[-3,3],
		basis=pyclups.basis.Hermite(3),
		curve=exper,
		output ="even_integrable",
		description="Even \& Integrable EIE",
		xnoise=0,
		sampling_rate = low
	)
if key == 6:
	constraint = pyclups.constraint.PositiveIntegrable(1.0)
	pyclups.analysis.RunTest(
		constraint=constraint,
		resolution=75,
		bounds=[-3,3],
		basis=pyclups.basis.Hermite(3),
		curve=exper,
		output ="positive_integrable",
		description="Positive \& Integrable EIE",
		xnoise=0,
		sampling_rate = low
	)