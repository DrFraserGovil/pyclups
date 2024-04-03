#pragma once
#include <vector>
#include "../customTypes.h"
#include "../Data/Data.h"
namespace cyclups
{
	class Prediction
	{
		public:
			Prediction(std::vector<double> x, std::vector<double> clups,std::vector<double> blup, std::vector<double> blp);
			PairedData CLUPS();
			PairedData BLUP();
			PairedData BLP();

			std::vector<double> X;
			std::vector<double> Y;
			std::vector<double> Y_BLUP;
			std::vector<double> Y_BLP;
		private:
	};	

	double TrueError(PairedData prediction, functionPointer f);

}