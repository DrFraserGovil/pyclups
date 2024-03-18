#include "basis.h"


namespace cyclups::basis
{
	Basis::Basis(int order, basisPointer bp): MaxOrder(order), Function(bp)
	{
	}
	double Basis::operator()(int order, double x)
	{
		if (order > MaxOrder)
		{
			return 0;
		}
		return Function(order,x);
	}

	Vector Basis::GetVector(double t)
	{
		Vector out(MaxOrder+1);
		for (int i = 0; i <= MaxOrder; ++i)
		{
			out[i] = Function(i,t);
		}
		return out;
	}

	double polyBasis(int i, double x)
	{
		return pow(x,i);
	}

	std::vector<std::vector<double>> HermiteParams = {{1.0},{0.,2.}};
	void GenerateHermite(int order)
	{
		if (order >= HermiteParams.size())
		{
			for (int j = HermiteParams.size(); j < order+1; ++j)
			{
				std::vector<double> as(j+1);
				as[0] = - HermiteParams[j-1][1];
				
				for (int k = 1; k <=j; ++k)
				{
					as[k] = 2*HermiteParams[j-1][k-1];
					if (k +1< HermiteParams[j-1].size())
					{
						as[k] -= (k+1)*HermiteParams[j-1][k+1];
					} 
				}
				HermiteParams.push_back(as);
			}
		}
	}

	double hermiteBasis(int i, double x)
	{

		double v = 0;
		for (int k = 0; k <= i; ++k)
		{
			v += HermiteParams[i][k] * pow(x,i);
		}
		return v;
	}


	Basis Polynomial(int order)
	{
		return Basis(order,polyBasis);
	}
	Basis Hermite(int order)
	{
		GenerateHermite(order);
		return Basis(order, hermiteBasis);
	}
}