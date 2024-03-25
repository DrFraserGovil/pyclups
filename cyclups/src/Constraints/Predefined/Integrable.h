#pragma once
#include "../ConstraintSet.h"
namespace cyclups::constraint
{
	class Integrable: public Constraint
	{
		public:
			Integrable(double value)
			{
				TargetIntegral = value;
			}
		private:
			double TargetIntegral;

			InitialiseContainer Initialiser(const std::vector<double> & t)
			{
				Matrix B = Matrix::Constant(1,t.size(),1);
				B(0,0) = 0.5; B(0,t.size()-1) = 0.5;
				double dt = t[1] - t[0];
				ConstraintVector c = ConstraintVector::Constant(1,TargetIntegral/dt);
				return InitialiseContainer(c,B); 
			}
	};
}