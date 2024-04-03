#pragma once
#include "Unimodal.h"

namespace cyclups::constraint
{
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
					// return;
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
					double maxVal = std::max(1e-5,input.maxCoeff());
					output[0] = log(std::max(1e-3,input[0]));
					
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
						prev = std::min(maxVal,std::max(1e-5,prev * exp(qi * exp(output[i]))));
						// if (prev < 1e-8)
						// {
						// 	output[i] = -10;
						// }
					}output[output.size()-1] = T;
				};
				ConstraintVector c = ConstraintVector::Optimise(n,n+1,FullTransform(f,grad,inv));
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