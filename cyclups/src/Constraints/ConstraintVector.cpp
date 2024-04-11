#include "ConstraintVector.h"


namespace cyclups::constraint
{
	ConstraintVector::ConstraintVector()
	{
	
	}
	ConstraintVector ConstraintVector::Optimise(int cDimension, int wDimension,SeparableTransform F)
	{
		return ConstraintVector(cDimension,wDimension,F);
	}
	ConstraintVector ConstraintVector::Optimise(int cDimension, int wDimension, FullTransform F)
	{
		return ConstraintVector(cDimension,wDimension,F);
	}

	ConstraintVector ConstraintVector::Constant(int dimension, double value)
	{
		std::vector<double> filled(dimension,value);
		return ConstraintVector(dimension,filled);
	};

	ConstraintVector ConstraintVector::Constant(int dimension, std::vector<double> value)
	{
		return ConstraintVector(dimension,value);
	};


	Vector & ConstraintVector::Value()
	{
		if (TransformDimension > 0)
		{
			Transform(Psi,W,TransformParams);
			value = Xi + Psi;	
		}
		
		return value;
	}


	Matrix &  ConstraintVector::Gradient()
	{
		if (simpleGradient)
		{
			SimpleFunction.GradF(spoofgradient,W,TransformParams);
			gradient = spoofgradient.asDiagonal();
		}
		else
		{
			FullFunction.GradF(gradient,W,TransformParams);
		}
		return gradient;
	}

	void  ConstraintVector::Initialise(const Vector & c)
	{
		Optim_M = Vector::Zero(TransformDimension);
		Optim_V = Vector::Zero(TransformDimension);
		BoundStep = 0;
		Inverse(W,c,TransformParams);
		gradient = Matrix::Zero(TransformDimension,Dimension);
		if (spoofgradient.size() > 0)
		{
			spoofgradient = Vector::Zero(TransformDimension);
		}
	}

	void  ConstraintVector::Step(VectorSlice v, int l, const OptimiserProperties & op)
	{
		double c1 = 1.0/(1.0 - pow(op.b1,l+1));
		double c2 = 1.0/(1.0 - pow(op.b2,l+1));
		double runSum = 0;
		for (int i = 0; i < TransformDimension; ++i)
		{
			double prev = W[i];
			double q = v[i];
			Optim_M[i] = op.b1 * Optim_M[i] + (1.0 - op.b1) * q;
			Optim_V[i] = op.b2 * Optim_V[i] + (1.0 - op.b2) * q*q;
			double denom = sqrt(Optim_V[i]*c2+1e-15);

			W[i] -= op.alpha * Optim_M[i]*c1 /denom;

			if (bounder)
			{
				++BoundStep;
				if (BoundStep > 1)
				{
					W[i] = std::min(maxWVal,std::max(W[i],minWVal));
				}
			}
		}
		
	}

	void  ConstraintVector::SetParams(std::vector<double> & params)
	{
		TransformParams = params;
	}
	void  ConstraintVector::SetBounds(double min, double max)
	{
		bounder = true;
		minWVal = min;
		maxWVal = max;
	}
	void  ConstraintVector::SavePosition()
	{
		BestW = W;
	}
	void  ConstraintVector::RecoverPosition()
	{
		W = BestW;
	}



	ConstraintVector::ConstraintVector(int dimension, std::vector<double> values) //simple constant constructor
	{
		IsConstant = true;

		Dimension = dimension;
		TransformDimension = 0;
		value = Vector::Map(&values[0],dimension);
	}
	ConstraintVector::ConstraintVector(int vectorDimension, int internalDimension, InvertableDifferentiableFunction<transformOperator,gradientOperator,transformOperator> F)
	{
		
		Dimension = vectorDimension;
		TransformDimension = internalDimension;
		FullFunction = F;
		Transform = F.F;
		Inverse = F.Inverse;
		simpleGradient = false;
		gradient = Matrix(internalDimension,vectorDimension);
		W = Vector(TransformDimension);
		Psi = Vector(Dimension);
		Xi = Vector::Zero(Dimension);
		Optim_M = Vector::Zero(TransformDimension);
		Optim_V = Vector::Zero(TransformDimension);
	}
	ConstraintVector::ConstraintVector(int vectorDimension, int internalDimension, InvertableDifferentiableFunction<transformOperator,transformOperator,transformOperator> F)
	{
		Dimension = vectorDimension;
		TransformDimension = internalDimension;
		
		// Transform = transform;
		// simpleDerivative = transformDerivative;
		// Inverse = inverse;
		SimpleFunction = F;
		Transform = F.F;
		Inverse = F.Inverse;
		simpleGradient = true;
		gradient = Matrix(internalDimension,vectorDimension);
		spoofgradient = Vector(vectorDimension);
		W = Vector(TransformDimension);
		Psi = Vector(TransformDimension);
		Xi = Vector::Zero(Dimension);
		Optim_M = Vector::Zero(TransformDimension);
		Optim_V = Vector::Zero(TransformDimension);
	}
}
