#pragma once
#include "dataArrays.h"
namespace cyclups::basis
{
	typedef double(* basisPointer)(int,double);
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