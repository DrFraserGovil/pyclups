#pragma once
#include <vector>
#include "JSL.h"

namespace cyclups
{
	//some simple datatypes for ferrying around XY coordinates in a recognisable manner

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