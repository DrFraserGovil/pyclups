#pragma once
#include <string>
#include <iostream>
#include <math.h>
namespace cyclups
{
class OptimiserProperties
	{
		public:
			bool Converged = false;
			double alpha = 0.6;
			double b1 = 0.7;
			double b2 = 0.9;
			double ConvergenceMemory = 0.99;
			int MaxSteps= 4000;
			int MinSteps = 0;

			double ConvergedGradient = 1e-3;
			double ConvergedScore = 1e-7;
			bool ReachedMaxSteps = false;
			bool GradientConverged = false;
			bool ScoreConverged = false;		
			
			OptimiserProperties(){};

			void Clear();
			
			void CheckConvergence(int l,double gradnorm);
			void CheckConvergence(int l, double gradnorm, double score);
			

			void PrintReason();
			

		private:
			double MaxAlpha=0;
			double MinAlpha;
			double PrevScore;
			double ScoreMemory;
			double GradientMemory;
			double triggeringGradient;
			double triggeringScore;
			int TriggeringStep;
			int NegativeCounter = 0;

			void UpdateAlpha(double score);
			
	};

}