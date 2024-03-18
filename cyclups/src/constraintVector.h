#pragma once
#include "dataArrays.h"
namespace cyclups::constraint
{

	struct VectorSlice
	{
		Vector & V;
		int SliceStart;
		int SliceSize;

		VectorSlice(Vector &v): V(v)
		{
			SliceStart = 0;
			SliceSize = v.size();
		}
		VectorSlice(Vector & v, int start,int size): V(v)
		
		{
			SliceStart = start;
			SliceSize = size;
		}
		double & operator[](int i)
		{
			return V[SliceStart + i];
		}
	};

	typedef void(* transformOperator)(Vector & output, const Vector & input,std::vector<double> & params);
	typedef void(* gradientOperator)(Matrix & output, const Vector & input,std::vector<double> & params);

	class ConstraintVector
	{
		public:

			int Dimension;
			int TransformDimension;
			bool IsConstant;
			

			ConstraintVector(){};
			
			static ConstraintVector Optimise(int cDimension, int wDimension,transformOperator transform, transformOperator transformDerivative, transformOperator inverse)
			{
				return ConstraintVector(cDimension,wDimension,transform,transformDerivative,inverse);
			}

			static ConstraintVector Constant(int dimension, double value)
			{
				std::vector<double> filled(dimension,value);
				return ConstraintVector(dimension,filled);
			};
			static ConstraintVector Constant(int dimension, std::vector<double> value)
			{
				return ConstraintVector(dimension,value);
			};

			Vector & Value()
			{
				if (TransformDimension > 0)
				{
					Transform(Psi,W,TransformParams);
					
					value = Xi + Psi;
				}
				return value;
			}

			Matrix & Gradient()
			{
				if (simpleGradient)
				{
					simpleDerivative(spoofgradient,W,TransformParams);
					gradient = spoofgradient.asDiagonal();
				}
				else
				{
					derivative(gradient,W,TransformParams);
				}
				return gradient;
			}

			void Initialise(const Vector & c)
			{
				Inverse(W,c,TransformParams);
			}

			void Step(VectorSlice v, int l, double alpha,double b1, double b2)
			{
				double c1 = 1.0/(1 - pow(b1,l+1));
				double c2 = 1.0/(1.0 - pow(b2,l+1));
				for (int i = 0; i < TransformDimension; ++i)
				{
					double q = v[i];
					Optim_M[i] = b1 * Optim_M[i] + (1.0 - b1) * q;
					Optim_V[i] = b2 * Optim_V[i] + (1.0 - b2) * q*q;
					double denom = sqrt(Optim_V[i] * c2 + 1e-16);
					W[i] -= alpha * c1 * Optim_M[i] /denom;

					if (bounder)
					{
						W[i] = std::min(maxWVal,std::max(W[i],minWVal));
					}
				}
			}

			void SetParams(std::vector<double> & params)
			{
				TransformParams = params;
			}
			void SetBounds(double min, double max)
			{
				bounder = true;
				minWVal = min;
				maxWVal = max;
			}
		private:
			Vector value;
			bool bounder = false;
			double minWVal;
			double maxWVal;
			ConstraintVector(int dimension, std::vector<double> values) //simple constant constructor
			{
				IsConstant = true;

				Dimension = dimension;
				TransformDimension = 0;
				value = Vector::Map(&values[0],dimension);
			}

			ConstraintVector(int vectorDimension, int internalDimension, transformOperator transform, transformOperator transformDerivative, transformOperator inverse)
			{
				Dimension = vectorDimension;
				TransformDimension = internalDimension;
				
				Transform = transform;
				simpleDerivative = transformDerivative;
				Inverse = inverse;
				simpleGradient = true;
				gradient = Matrix(vectorDimension,internalDimension);
				spoofgradient = Vector(vectorDimension);
				W = Vector(TransformDimension);
				Psi = Vector(TransformDimension);
				Xi = Vector::Zero(Dimension);
				Optim_M = Vector::Zero(TransformDimension);
				Optim_V = Vector::Zero(TransformDimension);
			}
			ConstraintVector(int vectorDimension, int internalDimension, transformOperator transform, gradientOperator transformDerivative, transformOperator inverse)
			{
				
				Dimension = vectorDimension;
				TransformDimension = internalDimension;
				Transform = transform;
				derivative = transformDerivative;
				Inverse = inverse;
				simpleGradient = false;
				gradient = Matrix(vectorDimension,internalDimension);
				W = Vector(TransformDimension);
				Psi = Vector(TransformDimension);
				Xi = Vector::Zero(Dimension);
				Optim_M = Vector::Zero(TransformDimension);
				Optim_V = Vector::Zero(TransformDimension);
			}
			std::vector<double> TransformParams;
			transformOperator Transform;
			transformOperator simpleDerivative;
			gradientOperator derivative;
			transformOperator Inverse;
			bool simpleGradient = false;
			Vector Xi;
			Vector Psi;
			Matrix gradient;
			Vector spoofgradient;
			Vector W;
			Vector Optim_M;
			Vector Optim_V;
	};


}