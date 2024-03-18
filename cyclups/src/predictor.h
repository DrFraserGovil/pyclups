#pragma once
#include "basis.h"
#include "kernel.h"
#include "prediction.h"
#include "Eigen"
#include "constraint.h"
namespace cyclups
{
	typedef const std::vector<double> &  cvec;


	struct Container
	{
		// Vector X;
		std::vector<Vector> a_blups;
		std::vector<double> p_blps;
		std::vector<double> p_blups;
		std::vector<Vector> ks;
		Vector Delta;
		double Beta;
		Vector Bp_blups;
		Eigen::LDLT<Matrix> BBt; 
		Matrix K;
	};

	class Predictor
	{
		public:
			
			template<class T>
			Predictor(kernel::Kernel k, basis::Basis b, T &constraint) : Kernel(k), Basis(b), Constraint(constraint){
			};

			// Predictor(kernel::Kernel k, basis::Basis b, constraint::ConstraintSet & constraint) : Kernel(k), Basis(b){
			// 	std::cout << "Specialised" << std::endl;

			// };
			
			// template<>
			// Predictor(kernel::Kernel k, basis::Basis b, constraint::ConstraintSet & constraint);
			Prediction Predict(cvec predictX, const PairedData & data, cvec dataErrors);
			Prediction Predict(cvec predictX,const PairedData & data, double dataErrors);

			void Retire();
		private:
			Container Store;
			kernel::Kernel Kernel;
			basis::Basis Basis;
			constraint::ConstraintSet Constraint;
			void Initialise(cvec PredictX, const PairedData & data, cvec dataErrors);

			void Optimise(cvec predictX, const PairedData & data, cvec dataErrors);
			double ComputeScore(cvec predictX);
	};
}