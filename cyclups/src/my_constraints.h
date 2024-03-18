#pragma once
#include "dataArrays.h"
#include "constraint.h"
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

			ConstraintContainer Initialiser(const std::vector<double> & t)
			{
				Matrix B = Matrix::Constant(1,t.size(),1);
				B(0,0) = 0.5; B(0,t.size()-1) = 0.5;
				double dt = t[1] - t[0];
				ConstraintVector c = ConstraintVector::Constant(1,TargetIntegral/dt);
				return ConstraintContainer(c,B); 
			}
	};


	class GreaterThan : public Constraint
	{
		public:
			GreaterThan(double value)
			{
				usingConst = true;
				constVal = value;
			}
			GreaterThan(double (* fnc)(double) )
			{
				usingFunction = true;
				fncVal = fnc;
			}
			GreaterThan(double value, bool (* domain)(double))
			{
				usingConst = true;
				constVal = value;
				usingDomain = true;
				inDomain = domain;
			}
			GreaterThan(double (* fnc)(double), bool (* domain)(double))
			{
				usingFunction = true;
				fncVal = fnc;
				usingDomain = true;
				inDomain = domain;
			}
		private:
			
			double (* fncVal)(double);
			bool usingFunction = false;
			bool usingConst = false;
			double constVal;
			

			ConstraintContainer Initialiser(cvec t)
			{
				cvec domainT = ApplyDomain(t);

				int n = domainT.size();
				Matrix B;
				if (domainT.size() == t.size())
				{
					B = Matrix::Identity(n,n);
				}
				else
				{
					B = Matrix::Zero(n,t.size());
					int q = 0;
					for (int i = 0; i < t.size(); ++i)
					{
						if (q< domainT.size() && t[i] == domainT[q])
						{
							B(q,i) = 1;
							++q;
						}
					}
				}

				transformOperator f = [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
					output[i] = params[i] + exp(input[i]);
				}};
				transformOperator grad= [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
					output[i] = exp(input[i]);
				}};
				transformOperator inv = [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
					double buffered = std::max(input[i] - params[i],1e-8);
					output[i] = log(buffered);
					std::cout << "GT  " << input[i] << "  " << params[i] + buffered << std::endl;
				}};
				ConstraintVector c = ConstraintVector::Optimise(n,n,f,grad,inv);
				if (usingConst)
				{
					std::vector<double> v (n,constVal);
					c.SetParams(v);
				}
				if (usingFunction)
				{
					std::vector<double> v(n);
					for (int i = 0; i < n; ++i)
					{
						v[i] = fncVal(domainT[i]);
					}
					c.SetParams(v);
				}
				return ConstraintContainer(c,B); 
			}
	};

	inline GreaterThan Positive()
	{
		return GreaterThan(0.); //has to be 0. as just 0 is the Null pointer which causes ambiguous conversion with the function pointer option
	}
	inline GreaterThan Positive(bool (*domain)(double))
	{
		return GreaterThan(0.,domain); //has to be 0. as just 0 is the Null pointer which causes ambiguous conversion with the function pointer option
	}

	class LessThan : public Constraint
	{
		public:
			LessThan(double value)
			{
				usingConst = true;
				constVal = value;
			}
			LessThan(double (* fnc)(double) )
			{
				usingFunction = true;
				fncVal = fnc;
			}
			LessThan(double value, bool (* domain)(double))
			{
				usingConst = true;
				constVal = value;
				usingDomain = true;
				inDomain = domain;
			}
			LessThan(double (* fnc)(double), bool (* domain)(double))
			{
				usingFunction = true;
				fncVal = fnc;
				usingDomain = true;
				inDomain = domain;
			}
		private:
			
			double (* fncVal)(double);
			bool usingFunction = false;
			bool usingConst = false;
			double constVal;
			
			

			ConstraintContainer Initialiser(cvec t)
			{
				cvec domainT = ApplyDomain(t);

				int n = domainT.size();
				Matrix B;
				if (domainT.size() == t.size())
				{
					B = Matrix::Identity(n,n);
				}
				else
				{
					B = Matrix::Zero(n,t.size());
					int q = 0;
					for (int i = 0; i < t.size(); ++i)
					{
						if (q< domainT.size() && t[i] == domainT[q])
						{
							B(q,i) = 1;
							++q;
						}
					}
				}

				transformOperator f = [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
					output[i] = params[i] - exp(input[i]);
				}};
				transformOperator grad= [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
					output[i] = - exp(input[i]);
				}};
				transformOperator inv = [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
					double buffered = std::max(-input[i] + params[i],1e-8);
					output[i] = log(buffered);
						std::cout << "LT " << input[i] << "  " << params[i] + buffered << std::endl;
				}};
				ConstraintVector c = ConstraintVector::Optimise(n,n,f,grad,inv);
				// c.SetBounds(-3,10);
				if (usingConst)
				{
					std::vector<double> v (n,constVal);
					std::cout << constVal << std::endl;
					c.SetParams(v);
				}
				if (usingFunction)
				{
					std::vector<double> v(n);
					for (int i = 0; i < n; ++i)
					{
						v[i] = fncVal(domainT[i]);
					}
					c.SetParams(v);
				}
				return ConstraintContainer(c,B); 
			}
	};

	inline LessThan Negative()
	{
		return LessThan(0.); //has to be 0. as just 0 is the Null pointer which causes ambiguous conversion with the function pointer option
	}
	inline LessThan Negative(bool (*domain)(double))
	{
		return LessThan(0.,domain); //has to be 0. as just 0 is the Null pointer which causes ambiguous conversion with the function pointer option
	}
	

}