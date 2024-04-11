#pragma once
#include "../ConstraintSet.h"
namespace cyclups::constraint
{
	namespace direction
	{
		enum GradientDirection {Positive,Negative};
	}
	class SteeperThan : public Constraint
	{
		public:
			SteeperThan(double value)
			{
				Direction = direction::Positive;
				constVal = value;
			}
			SteeperThan(direction::GradientDirection dir, double value)
			{
				constVal = value;
				Direction = dir;
			}

			SteeperThan(double value, bool (* domain)(double))
			{
				Direction = direction::Positive;
				constVal = value;
				usingDomain = true;
				inDomain = domain;
			}

			SteeperThan(direction::GradientDirection dir, double value, bool (* domain)(double))
			{
				Direction = dir;
				constVal = value;
				usingDomain = true;
				inDomain = domain;
			}

		private:
			direction::GradientDirection Direction;
			double constVal;

			InitialiseContainer Initialiser(cvec t)
			{
				cvec domainT = ApplyDomain(t);

				int n = domainT.size();
				Matrix B = Matrix::Zero(n-1,t.size());
				

				for (int i = 1; i < n; ++i)
				{
					for (int ri = 1; ri < t.size(); ++ri)
					{
						if (t[ri] == domainT[i])
						{
							B(i-1,ri-1) = -1;
							B(i-1,ri) = 1;
						}
					}
				}
				transformOperator f;
				transformOperator grad;
				transformOperator inv;

				if (Direction == direction::Positive)
				{
					f = [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
						output[i] = params[i] + exp(input[i]);
					}};
					grad= [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
						output[i] = exp(input[i]);
					}};
					inv = [](Vector & output, const Vector & input,std::vector<double> & params){
						for (int i =0; i < output.size(); ++i){
						double buffered = std::max(input[i] - params[i],1e-4);
						output[i] = log(buffered);
					}};
				}
				else
				{
					f = [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
						output[i] = params[i] - exp(input[i]);
					}};
					grad= [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
						output[i] = -exp(input[i]);
					}};
					inv = [](Vector & output, const Vector & input,std::vector<double> & params){
						for (int i =0; i < output.size(); ++i){
						double buffered = std::max(params[i] - input[i],1e-4);
						output[i] = log(buffered);
					}};
				}
				ConstraintVector c = ConstraintVector::Optimise(n-1,n-1,SeparableTransform(f,grad,inv));
				
				std::vector<double> v (n,constVal);
				c.SetParams(v);
				
				return InitialiseContainer(c,B); 
			}
	};
	inline SteeperThan Monotonic()
	{
		return SteeperThan(0.); //has to be 0. as just 0 is the Null pointer which causes ambiguous conversion with the function pointer option
	}
	inline SteeperThan Monotonic(bool (*domain)(double))
	{
		return SteeperThan(0.,domain); //has to be 0. as just 0 is the Null pointer which causes ambiguous conversion with the function pointer option
	}
	inline SteeperThan Monotonic(direction::GradientDirection dir)
	{
		return SteeperThan(dir,0.); 
	}
	inline SteeperThan Monotonic(direction::GradientDirection dir,bool (*domain)(double))
	{
		return SteeperThan(dir,0.,domain); 
	}
}


