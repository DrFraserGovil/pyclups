#pragma once
#include "predictor.h"
#include "../Data/Data.h"
#include "JSL.h"
#include <sstream>
// extern JSL::gnuplot gpu;
namespace cyclups
{
	void Curve(JSL::gnuplot & gp, cyclups::PairedData curve, std::string name, double (*func)(double));
	class EIE
	{
		public:
			int minData = 8;
			int maxData = 20;
			int PredictionResolution = 100;
			double PredictLowerBound = -3;
			double PredictUpperBound = 3;

			double noiseMin = 0.01;
			double noiseMax = 0.1;
			int noiseResolution = 50;

			double scaleMin = 0.1;
			double scaleMax = 2;
			int scaleResolution = 50;
			EIE(){};

			template<class T>
			void Run(functionPointer curve, T constraint,  kernel::Kernel K, basis::Basis B, int Resolution,std::string outfile )
			{
				constraint::ConstraintSet con(constraint);
				InternalRun(curve,con, K, B, Resolution, outfile);
			}

			void Recover(functionPointer curve, constraint::ConstraintSet constraint, kernel::Kernel K, basis::Basis B,int samples,int id, int seed);

			void Plot(std::string file, std::string name);
		private:

			void InternalRun(functionPointer curve, constraint::ConstraintSet constraint, kernel::Kernel K, basis::Basis B, int Resolution,std::string outfile);

	};
}