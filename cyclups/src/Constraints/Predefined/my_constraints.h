#pragma once
#include "../../customTypes.h"
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
			

			InitialiseContainer Initialiser(cvec t)
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
				return InitialiseContainer(c,B); 
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
			
			

			InitialiseContainer Initialiser(cvec t)
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
				}};
				ConstraintVector c = ConstraintVector::Optimise(n,n,f,grad,inv);
				// c.SetBounds(-3,10);
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
				return InitialiseContainer(c,B); 
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
	


	class BoundedBetween: public Constraint
	{
		public:
			BoundedBetween(double value1, double value2)
			{
				usingConst = true;
				lowerVal = std::min(value1,value2);
				upperVal = std::max(value1,value2);
			}
			BoundedBetween(double value1, double value2,bool (* domain)(double)): BoundedBetween(value1,value2)
			{
				setDomain(domain);
			}
			BoundedBetween(double (* f1)(double),double (* f2)(double))
			{
				usingFunction= true;
				lowerFunc = f1;
				upperFunc = f2;
			}
			BoundedBetween(double (* f1)(double),double (* f2)(double), bool (*domain)(double)) : BoundedBetween(f1,f2)
			{
				setDomain(domain);
			}
			BoundedBetween(double value1, double (*f)(double))
			{
				usingMixed= true;
				lowerVal = value1;
				upperFunc= f; //ordering not actually important here
			}
			BoundedBetween(double value1, double (*f)(double),bool (*domain)(double)) : BoundedBetween(value1,f)
			{
				setDomain(domain);
			}

		private:
			void setDomain(bool (* domain)(double))
			{
				usingDomain = true;
				inDomain = domain;
			}
			bool usingMixed = false;
			double (* lowerFunc)(double);
			double (* upperFunc)(double);
			bool usingFunction = false;
			bool usingConst = false;
			double lowerVal;
			double upperVal;

			InitialiseContainer Initialiser(cvec t)
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
					output[i] = params[2*i] + (params[2*i+1] - params[2*i])/(1.0 + exp(-input[i]));
				}};
				transformOperator grad= [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){
					output[i] = exp(-input[i]) *  (params[2*i+1] - params[2*i]) / pow((1.0 + exp(-input[i])),2);
				}};
				transformOperator inv = [](Vector & output, const Vector & input,std::vector<double> & params){for (int i =0; i < output.size(); ++i){

					double buffered = std::max(params[2*i] + 1e-8,std::min(params[2*i+1] - 1e-8,input[i]));
					double frac = (buffered - params[2*i])/(params[2*i+1] - buffered);
					output[i] = log(frac);
				}};
				ConstraintVector c = ConstraintVector::Optimise(n,n,f,grad,inv);
				if (usingConst)
				{
					std::vector<double> v (2*n);
					for (int i =0; i < n; ++i)
					{
						v[2*i] = lowerVal;
						v[2*i+1] = upperVal;
					}
					c.SetParams(v);
				}
				// if (usingFunction)
				// {
				// 	std::vector<double> v(n);
				// 	for (int i = 0; i < n; ++i)
				// 	{
				// 		v[i] = fncVal(domainT[i]);
				// 	}
				// 	c.SetParams(v);
				// }
				return InitialiseContainer(c,B); 
			}
	};



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
				}output[output.size()-1] = 4;
				};
				ConstraintVector c = ConstraintVector::Optimise(n,n+1,f,grad,inv);
				std::vector<double> v = t;
				// v.push_back(t[1] )
				c.SetParams(v);
				//  c.SetBounds(-12,12);
				return InitialiseContainer(c,B); 
			}
	};

	class PositiveUnimodal: public Constraint
	{
		public:
			PositiveUnimodal(){};

		private:
			InitialiseContainer Initialiser(cvec t)
			{
				int n = t.size();
				Matrix B = Matrix::Identity(n,n);
				// std::cout << B << std::endl;



				transformOperator f = [](Vector & output, const Vector & input,std::vector<double> & params){
					int n = input.size() - 1;
					output[0] = exp(input[0]);
					double T = input[n];
					double delta = params[1] - params[0];
					for (int i =1; i < n; ++i)
					{
						double mult = exp(smoothQ(params[i],T,delta) * exp(input[i]));
						output[i] = mult * output[i-1];
						// std::cout << i << "  " << params[i] << "  "  << input[i] << "  " << mult << "  " << output[i] << std::endl; 
					}
				};
				gradientOperator grad= [](Matrix & output, const Vector & input,std::vector<double> & params){
					int n = input.size() - 1;
					
					//compute f_i along the diagonals
					output(0,0) = exp(input[0]);
					double T = input[n];
					double delta = params[1] - params[0];
					double prevFi = 0;
					double runSum = 0;
					for (int i = 0; i < n; ++i)
					{
						double fi;
						if (i == 0)
						{
							fi = exp(input[0]);
						}
						else
						{
							double mult = exp(smoothQ(params[i],T,delta) * exp(input[i]));
							fi = mult * prevFi;
						}
						for (int j = 0; j<=i; ++j)
						{
							//d f_i / d w_j
							if (j == 0)
							{
								output(j,i) = fi * exp(input[j]);
							}
							else
							{
								output(j,i) = fi * smoothQ(params[j],T,delta) * exp(input[j]);
							}
						}
						if (i > 0)
						{
							double mySum = deltaSmoothQ(params[i],input[n],params[1] - params[0])*exp(input[i]);
							runSum += mySum;
							output(n,i) = fi * runSum;
						}
						prevFi = fi;
					}
					
					// int n = input.size() - 1;
					// for (int i =0; i < n; ++i)
					// {
					// 	output(i,i) = exp(input[i])* (smoothQ(params[i],input[n],params[1] - params[0]) + deltaSmoothQ(params[i+1],input[n],params[1] - params[0]));
					// 	output(n,i) = deltaSmoothQ(params[i],input[n],params[1] - params[0])*exp(input[i]);
					// }
					// std::cout<< output << std::endl;
					// for (int j = 0; j < n; ++j)
					// output(n,n) = -(input[n]-3);
				};
				transformOperator inv = [](Vector & output, const Vector & input,std::vector<double> & params){
					// double T = params[params.size()-1];
					// if (T < params[0] || T > params[params.size()-2])
					// {
					// 	T = params[params.size()/2];
					// }
					double delta = params[1] - params[0];
					double prev=1;

					Eigen::MatrixXf::Index max_index;
					input.maxCoeff(&max_index);
					double T = params[(int)max_index];
					params[params.size()-1] = T;
					double maxVal = std::max(1e-7,input.maxCoeff());
					output[0] = log(std::max(1e-8,input[0]));
					
					prev = exp(output[0]);

					double stepsToAnsatz  =(T -params[0])/delta-1;
					double scaleHeight = log(abs(log(prev/maxVal))/stepsToAnsatz);

					for (int i =1; i < input.size(); ++i)
					{
						// if (i == 0)
						// {
						// 	output[i] = log(std::max(input[i],1e-8));
						// 	prev = exp(output[i]);
						// }
						// else
						// {
							
						// 	double exper = std::max(1e-15,input[i]/prev);
						// 	double arg = std::max(1e-15,log(exper)/(qi+1e-10));
						// 	output[i] = log(std::max(1e-15,arg));

						// 	output[i] = std::min(5.,std::max(-5.,output[i]));
							
						// 	prev = std::max(1e-10,prev * exp(qi * exp(output[i])));
						// 	// std::cout << "prev " << i << "  " << qi << "  " << prev << "  " << output[i] << "  " << input[i] <<  std::endl;
						// }
						
						// double q = smoothQ(params[i],T,delta);
						double qi = smoothQ(params[i],T,delta);
						double scale = log(log(params.size())+8);

						output[i] = scaleHeight;
						prev = std::max(1e-8,prev * exp(qi * exp(output[i])));
						// if (prev < 1e-8)
						// {
						// 	output[i] = -10;
						// }
					}output[output.size()-1] = T;
				};
				ConstraintVector c = ConstraintVector::Optimise(n,n+1,f,grad,inv);
				std::vector<double> v = t;
				// v.resize(v.size()*2);
				// v.push_back(t[1] )
				v.push_back(0);
				c.SetParams(v);
				//  c.SetBounds(-1,1);
				return InitialiseContainer(c,B); 
			}
	};
}