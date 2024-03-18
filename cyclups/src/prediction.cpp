#include "prediction.h"

namespace cyclups
{
	double TrueError(PairedData prediction, functionPointer f)
	{
		double s = 0;
		int n = prediction.X.size();
		for (int i = 0; i < n;  ++i)
		{
			double truth = f(prediction.X[i]);
			s+= pow(truth - prediction.Y[i],2);
		}
		return sqrt(s/n);
	}

	cyclups::Prediction::Prediction(std::vector<double> x, std::vector<double> clups, std::vector<double> blup, std::vector<double> blp)
	{
		X = x;
		Y = clups;
		Y_BLP = blp;
		Y_BLUP = blup;
	}

	PairedData cyclups::Prediction::CLUPS()
	{
		return PairedData(X,Y);
	}

	PairedData cyclups::Prediction::BLUP()
	{
		return PairedData(X,Y_BLUP);
	}

	PairedData cyclups::Prediction::BLP()
	{
		return PairedData(X,Y_BLP);
	}
}