#pragma once
#include "dataArrays.h"
#include "constraintVector.h"
#include <functional>
namespace cyclups::constraint
{
	class ConstraintSet;

	struct ConstraintContainer
	{
		bool Initialised;
		ConstraintVector vector;
		Matrix matrix;
		ConstraintContainer()
		{
			Initialised = false;
		}
		ConstraintContainer(ConstraintVector &v,Matrix & m)
		{
			vector = v;
			matrix = m;
			Initialised = true;
		}

	};

	class Constraint
	{
		public:
			int Dimension;
			int TransformDimension;
			bool IsConstant;

			Constraint(){};
			Constraint(Matrix mat,ConstraintVector vec)
			{
				Dimension = vec.Dimension;
				IsConstant = true;
				TransformDimension = 0;
				vector = vec;
			};

			

			Matrix matrix;
			ConstraintVector vector;

			void Initialise(const std::vector<double> & ts)
			{
				ConstraintContainer init = Initialiser(ts);
				if (init.Initialised)
				{
					vector = init.vector;
					matrix = init.matrix;
					Dimension = vector.Dimension;
					TransformDimension = vector.TransformDimension;
					IsConstant = true;
					if (TransformDimension > 0)
					{
						IsConstant = false;
					}

				}
			}

		protected:
			bool usingDomain = false;
			bool (* inDomain)(double);
			
			virtual ConstraintContainer Initialiser(const std::vector<double> & t)
			{
				return ConstraintContainer();
			}

			std::vector<double> ApplyDomain(const std::vector<double> & t)
			{
				if (!usingDomain)
				{
					return t;
				}
				else
				{
					std::vector<double> tt;
					for (int i = 0; i < t.size(); ++i)
					{
						if (inDomain(t[i]))
						{
							tt.push_back(t[i]);
						}
					}
					return tt;
				}
			} 

	
			
	};


	
	




}