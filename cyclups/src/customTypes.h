#pragma once
#include "Eigen"

namespace cyclups
{
	typedef double(* functionPointer)(double);
	typedef double(* basisPointer)(int,double);

	typedef Eigen::VectorXd Vector;
	typedef Eigen::MatrixXd Matrix;
}