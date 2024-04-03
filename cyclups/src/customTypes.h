#pragma once
#include "Eigen"

namespace cyclups
{
	typedef double(* functionPointer)(double);
	typedef double(* basisPointer)(int,double);


	typedef const std::vector<double> &  cvec;
	typedef Eigen::VectorXd Vector;
	typedef Eigen::MatrixXd Matrix;
	typedef double(* vectorScalarFunction)(const Vector & input);
	typedef void(* vectorVectorFunction)(Vector & output,const Vector & input);
	typedef void(* transformOperator)(Vector & output, const Vector & input,std::vector<double> & params);
	typedef void(* gradientOperator)(Matrix & output, const Vector & input,std::vector<double> & params);

	template<class T, class S>
	struct DifferentiableFunction
	{
		DifferentiableFunction(){};
		DifferentiableFunction(T t, S s) : F(t), GradF(s) {};
			T F;
			S GradF;
	};
	template<class T, class S, class V>
	struct InvertableDifferentiableFunction
	{
		InvertableDifferentiableFunction(){};
		InvertableDifferentiableFunction(T t, S s, V v): F(t), GradF(s), Inverse(v) {};
		T F;
		S GradF;
		V Inverse;
	};	
	typedef InvertableDifferentiableFunction<transformOperator,transformOperator,transformOperator> SeparableTransform ;
	typedef InvertableDifferentiableFunction<transformOperator,gradientOperator,transformOperator> FullTransform ;

	typedef DifferentiableFunction<vectorScalarFunction,vectorVectorFunction> RegularisingFunction;
}