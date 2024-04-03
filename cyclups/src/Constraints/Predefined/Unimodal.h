#pragma once
#include "../ConstraintSet.h"
namespace cyclups::constraint
{

	inline double smoothQ(double t, double T,double delta)
	{
		delta/=3;
		double v =  -1.0 + 2.0/(1.0 + exp((t - T)/delta));
		return v;
	}
	inline double deltaSmoothQ(double t, double T,double delta)
	{
		delta/=3;
		double e = exp((t - T)/delta);
		return 2.0/pow(1.0 + e,2)*e/delta;
	}

	class Unimodal: public Constraint
	{
		public:
			Unimodal(){};

		private:
			InitialiseContainer Initialiser(cvec t)
			{
				int n = t.size() - 1;
				Matrix B = Matrix::Zero(n,t.size());
				for (int i = 0; i < n; ++i)
				{
					B(i,i) = -1;
					B(i,i+1) = 1;
				}
				// std::cout << B << std::endl;



				transformOperator f = [](Vector & output, const Vector & input,std::vector<double> & params){
					int n = input.size() - 1;
					for (int i =0; i < n; ++i){
					output[i] = exp(input[i]) * smoothQ(params[i+1],input[n],params[1] - params[0]);
				}};
				gradientOperator grad= [](Matrix & output, const Vector & input,std::vector<double> & params){
					int n = input.size() - 1;
					for (int i =0; i < n; ++i)
					{
						output(i,i) = exp(input[i])* (smoothQ(params[i+1],input[n],params[1] - params[0]) + deltaSmoothQ(params[i+1],input[n],params[1] - params[0]));
						output(n,i) = deltaSmoothQ(params[i],input[n],params[1] - params[0])*exp(input[i]);
					}
					// std::cout<< output << std::endl;
					// for (int j = 0; j < n; ++j)
					// output(n,n) = -(input[n]-3);
				};
				transformOperator inv = [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
					// double buffered = std::max(input[i],1e-8);
					output[i] = log(0.02);
				}output[output.size()-1] = 0;
				};
				ConstraintVector c = ConstraintVector::Optimise(n,n+1,FullTransform(f,grad,inv));
				std::vector<double> v = t;
				// v.push_back(t[1] )
				c.SetParams(v);
				//  c.SetBounds(-12,12);
				return InitialiseContainer(c,B); 
			}
	};
}