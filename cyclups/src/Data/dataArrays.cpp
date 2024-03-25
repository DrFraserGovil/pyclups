#include "dataArrays.h"

namespace cyclups
{
	

	PairedData::PairedData(int n)
	{
		X.resize(n,0);
		Y.resize(n,0);
	}

	PairedData::PairedData(std::vector<double> x, std::vector<double> y)
	{
		if (x.size() != y.size())
		{
			JSL::Error("Paired Vectors must be the same size");
		}
		X = x;
		Y = y;
	}

	Pair PairedData::operator[](int i) const
	{
		return Pair(X[i],Y[i]);
	}

	std::vector<Pair> PairedData::GetPairs()
	{
		std::vector<Pair> out;
		for (int i = 0; i < X.size(); ++i)
		{
			out.push_back(Pair(X[i],Y[i]));
		}
	}
}
