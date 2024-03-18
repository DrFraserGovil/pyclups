#pragma once
#include "dataArrays.h"
#include "Eigen"
namespace cyclups::kernel
{
	typedef double(* kernelFunctionPointer)(double,double, std::vector<double>);
	class Kernel
	{
		public:
			Kernel(kernelFunctionPointer f,std::vector<double> params);

			double operator()(double x, double y);

			Vector GetVector(double predictT, const std::vector<double> & dataT,const std::vector<double> & dataVariance);
			Matrix GetMatrix(const std::vector<double> &dataT,const std::vector<double> & dataVariance);
		private:
			const kernelFunctionPointer function;
			std::vector<double> Parameters;
	};



	Kernel SquaredExponential(double signalVariance, double lengthScale);

	Kernel Exponential(double signalVariance, double lengthScale);
}