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
			double alpha = 0.1;
			double b1 = 0.8;
			double b2 = 0.9;
			double ConvergenceMemory = 0.99;
			int MaxSteps= 1500;
			int MinSteps = 5;

			double ConvergedGradient = 1e-8;
			double ConvergedScore = 1e-10;
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