#pragma once
#include "../customTypes.h"
#include "VectorSlice.h"
#include "../Engine/OptimiserProperties.h"
namespace cyclups::constraint
{

	class ConstraintVector
	{
		public:

			int Dimension;
			int TransformDimension;
			bool IsConstant;
			

			ConstraintVector();
			
			//pseudo-constructors -- replacement for the pythonic subclassing shenanigans
			static ConstraintVector Optimise(int cDimension, int wDimension,SeparableTransform F);
			static ConstraintVector Optimise(int cDimension, int wDimension,FullTransform F);

			static ConstraintVector Constant(int dimension, double value);
			static ConstraintVector Constant(int dimension, std::vector<double> value);

			Vector & Value();

			Matrix & Gradient();

			void Initialise(const Vector & c);

			void Step(VectorSlice v, int l, const OptimiserProperties & op);

			void SetParams(std::vector<double> & params);
			void SetBounds(double min, double max);
			void SavePosition();
			void RecoverPosition();
		private:
			Vector value;
			bool bounder = false;
			double minWVal;
			double maxWVal;
			int BoundStep = 0;
			std::vector<double> TransformParams;
			transformOperator Transform;
			transformOperator Inverse;
			InvertableDifferentiableFunction<transformOperator,transformOperator,transformOperator> SimpleFunction;
			InvertableDifferentiableFunction<transformOperator,gradientOperator,transformOperator> FullFunction;
			bool simpleGradient = false;
			Vector Xi;
			Vector Psi;
			Matrix gradient;
			Vector spoofgradient;
			Vector W;
			Vector BestW;
			Vector Optim_M;
			Vector Optim_V;
			
			//simple constant constructor. Can only be called through the Constant static function
			ConstraintVector(int dimension, std::vector<double> values); 

			//optimising constructor with a linear gradient. Can only be called through the associated Optimise static function.
			ConstraintVector(int vectorDimension, int internalDimension, InvertableDifferentiableFunction<transformOperator,transformOperator,transformOperator> F);
			
			
			// transformOperator transform, transformOperator transformDerivative, transformOperator inverse);

			//optimising constructor with a full matrix gradient. Can only be called through the associated Optimise static function.
			ConstraintVector(int vectorDimension, int internalDimension, InvertableDifferentiableFunction<transformOperator,gradientOperator,transformOperator> F); 
			
	};


}