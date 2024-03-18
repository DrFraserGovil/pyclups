#pragma once
#include <vector>
#include "JSL.h"
#include "Eigen"
namespace cyclups
{
	typedef double(* functionPointer)(double);
	typedef Eigen::VectorXd Vector;
	typedef Eigen::MatrixXd Matrix;
	
	struct Pair
	{
		double X;
		double Y;
		Pair(double x, double y)
		{
			X = x;
			Y = y;
		}
	};

	class PairedData
	{
		public:
			std::vector<double> X;
			std::vector<double> Y;

			PairedData(int n);
			
			PairedData(std::vector<double> x, std::vector<double> y);

			Pair operator [](int i) const;
			std::vector<Pair> GetPairs();
	};
}