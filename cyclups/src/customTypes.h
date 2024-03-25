#pragma once
#include "Eigen"

namespace cyclups
{
	typedef double(* functionPointer)(double);
	typedef double(* basisPointer)(int,double);

	typedef const std::vector<double> &  cvec;
	typedef Eigen::VectorXd Vector;
	typedef Eigen::MatrixXd Matrix;

	typedef void(* transformOperator)(Vector & output, const Vector & input,std::vector<double> & params);
	typedef void(* gradientOperator)(Matrix & output, const Vector & input,std::vector<double> & params);
}