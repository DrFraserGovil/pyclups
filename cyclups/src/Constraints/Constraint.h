#pragma once
#include "../Data/Data.h"
#include "constraintVector.h"
#include <functional>
namespace cyclups::constraint
{
	struct InitialiseContainer
	{
		bool Initialised;
		ConstraintVector vector;
		Matrix matrix;
		InitialiseContainer();
		InitialiseContainer(ConstraintVector &v,Matrix & m);
	};

	class Constraint
	{
		public:
			Matrix matrix;
			ConstraintVector vector;

			int Dimension;
			int TransformDimension;
			bool IsConstant;

			Constraint();
			Constraint(Matrix mat,ConstraintVector vec);		

			void CallInitialiser(const std::vector<double> & ts);


		protected:
			bool usingDomain = false;
			bool (* inDomain)(double);
			
			virtual InitialiseContainer Initialiser(const std::vector<double> & t);
			std::vector<double> ApplyDomain(const std::vector<double> & t);
			
	};


	
	




}