#include "kernel.h"

cyclups::kernel::Kernel::Kernel(kernelFunctionPointer f,std::vector<double> params): function(f)
{
	Parameters = params;
}

double cyclups::kernel::Kernel::operator()(double x, double y)
{	
	return function(x,y,Parameters);
}

cyclups::Vector cyclups::kernel::Kernel::GetVector(double predictT, const std::vector<double> & dataT,const std::vector<double> & dataErrors)
{
	int n = dataT.size();
	cyclups::Vector out(n);

	for (int i = 0; i < n; ++i)
	{
		out[i] = function(predictT,dataT[i],Parameters);
	}
	return out;
}

cyclups::Matrix cyclups::kernel::Kernel::GetMatrix(const std::vector<double> & dataT,const std::vector<double> &dataErrors)
{
	int n = dataT.size();
	cyclups::Matrix out(n,n);

	for (int i = 0; i < n; ++i)
	{
		out(i,i) = function(dataT[i],dataT[i],Parameters) + dataErrors[i]*dataErrors[i];
		for (int j = i+1; j < n; ++j)
		{
			double val = function(dataT[j],dataT[i],Parameters);
			out(i,j) = val;
			out(j,i) = val;
		}
	}
	return out;
}

cyclups::kernel::Kernel cyclups::kernel::SquaredExponential(double signalVariance, double lengthScale)
{
	std::vector<double> p = {signalVariance,lengthScale};
	return cyclups::kernel::Kernel([](double x, double y,std::vector<double> params){return params[0] * exp(-0.5*pow((x-y)/params[1],2));},p);
}
cyclups::kernel::Kernel cyclups::kernel::Exponential(double signalVariance, double lengthScale)
{
	std::vector<double> p = {signalVariance,lengthScale};
	return cyclups::kernel::Kernel([](double x, double y,std::vector<double> params){return params[0] * exp(-abs((x-y)/params[1]));},p);
}
