#pragma once
#include "../customTypes.h"
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
			void UpdateParameter(std::vector<double> & newParam);
			void UpdateParameter(int index, double value);
		private:
			const kernelFunctionPointer function;
			std::vector<double> Parameters;
	};



	Kernel SquaredExponential(double lengthScale, double signalVariance);

	Kernel Exponential(double lengthScale, double signalVariance);
}