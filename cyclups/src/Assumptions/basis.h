#pragma once
#include "../customTypes.h"
namespace cyclups::basis
{
	
	class Basis
	{
		public:
			const int MaxOrder;
			Basis(int order, basisPointer bp);

			double operator()(int order, double x);
			Vector GetVector(double t);
		private:
			basisPointer Function;
	};



	extern std::vector<std::vector<double>> HermiteParams;
	Basis Polynomial(int order);
	Basis Hermite(int order);
}