#include "generator.h"

namespace cyclups
{


	namespace generator
	{
		std::default_random_engine randomiser;
		std::normal_distribution<double> gaussian (0.0,1.);
		std::uniform_real_distribution<double> uniform(0,1);
		PairedData generateSamples(std::vector<double> x, functionPointer f, double noise)
		{
			int n = x.size();
			std::vector<double> y(n);
			
			for (int i = 0; i < n; ++i)
			{
				y[i] = f(x[i]) + gaussian(randomiser) *noise;
			}
			return PairedData(x,y);
		}


		PairedData Sample(int N, functionPointer f, double xMin, double xMax, double noise)
		{
			return UniformXSample(N,f,xMin, xMax, noise);
		}

		PairedData UniformXSample(int N, functionPointer f, double xMin, double xMax, double noise)
		{
			std::vector<double> x = JSL::Vector::linspace(xMin,xMax,N);

			return generateSamples(x,f,noise);
		}

		PairedData RandomXSample(int N, functionPointer f, double xMin, double xMax, double noise)
		{
			std::vector<double> x = JSL::Vector::RandVec(N,xMin,xMax);
			std::sort(x.begin(),x.end());
			return generateSamples(x,f,noise);
		}
		PairedData NoisyXSample(int N, functionPointer f, double xMin, double xMax, double noise)
		{
			std::vector<double> x = JSL::Vector::linspace(xMin,xMax,N);
			double dx = (x[1] - x[0]);
			for (int i = 0; i < N; ++i)
			{
				x[i] += gaussian(randomiser) * dx/2;
			}
			std::sort(x.begin(),x.end());
			return generateSamples(x,f,noise);
		}
	}
}