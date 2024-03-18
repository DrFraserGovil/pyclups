#pragma once
#include <vector>
#include "subconstraint.h"
#include <stdexcept>
namespace cyclups::constraint
{
	
	class ConstraintSet
	{
		public:

			ConstraintSet(){Dimension = 0;};
			ConstraintSet(ConstraintSet & copy)
			{
				copy.PotencyCheck();
				PotencyCheck();
				Constraints = std::move(copy.Constraints);
				WasCopied = true;
				copy.Impotent = true;
				DataOrigin = &copy;
				
			}
			~ConstraintSet()
			{
				Retire();
			}
			template <class T>
			ConstraintSet(T s)
			{
				Add(s);
			}
			
			template<class T>
			void Add(T s)
			{
				auto y = std::make_unique<T>(s);
				Constraints.push_back(std::move(y));
				
			}
			void Add(ConstraintSet c)
			{
				PotencyCheck();
				c.PotencyCheck();
				Constraints = std::move(c.Constraints);
			}
			void Transfer(ConstraintSet * c)
			{
				c->Constraints = std::move(Constraints);
			}

			void Retire()
			{
				if (WasCopied)
				{
					Transfer(DataOrigin);
					DataOrigin->Impotent = false;
				}
			}

			void SetPosition(Vector c)
			{
				double start = 0;
				for (int i = 0; i < Constraints.size(); ++i)
				{
					int dim = Constraints[i]->Dimension;
					Vector subc = c.segment(start,dim);
					Constraints[i]->vector.Initialise(subc);
					start += dim;
				}
			}

			void Initialise(const std::vector<double> & t)
			{
				for (int i = 0; i < Constraints.size(); ++i)
				{
					Constraints[i]->Initialise(t);
				}
				Construct(t);
			}
			void Construct(std::vector<double> ts)
			{
				//can't add matrices together until size determined
				Dimension = 0;
				TransformDimension = 0;
				for (int i =0; i < Constraints.size(); ++i)
				{
					Dimension += Constraints[i]->Dimension;
					TransformDimension += Constraints[i]->TransformDimension;
				}
				IsConstant = (TransformDimension == 0);
				B = Matrix::Zero(Dimension,ts.size());
				c = Vector::Zero(Dimension);
				int r = 0;

				for (int i = 0; i < Constraints.size(); ++i)
				{
					Matrix & submat = Constraints[i]->matrix;
					Vector subvec = Constraints[i]->vector.Value();
					for (int j = 0; j < submat.rows(); ++j)
					{
						B.row(r) << submat.row(j);
						c[r] = subvec[j];
						++r;
					}
				}

				Grad = Matrix::Zero(TransformDimension,Dimension);
			}

			bool IsConstant = false;
			int Dimension;
			int TransformDimension;
			Matrix B;
			Matrix Grad;

			Vector VectorValue()
			{
				if (!IsConstant)
				{
					c = Vector::Zero(Dimension);
					int r = 0;
					for (int i = 0; i < Constraints.size(); ++i)
					{
						Matrix & submat = Constraints[i]->matrix;
						Vector subvec = Constraints[i]->vector.Value();
						for (int j = 0; j < submat.rows(); ++j)
						{
							B.row(r) << submat.row(j);
							c[r] = subvec[j];
							++r;
						}
					}
				}
				return c;
			}
			Matrix Gradient()
			{
				int c_Ind = 0;
				int w_Ind = 0;
				for (int i = 0; i < Constraints.size(); ++i)
				{
					int dim = Constraints[i]->Dimension;
					int wDim = Constraints[i]->TransformDimension;
					if (wDim > 0)
					{
						Grad.block(w_Ind,c_Ind,wDim,dim) << Constraints[i]->vector.Gradient();
					}
					c_Ind += dim;
					w_Ind += wDim;
				}
				return Grad;
			}
			void Step(Vector & grad,int l, double alpha,double b1, double b2)
			{
				int start = 0;
				for (int i = 0; i < Constraints.size(); ++i)
				{
					int dim = Constraints[i]->TransformDimension;
					auto v = VectorSlice(grad,start,dim);
					Constraints[i]->vector.Step(v,l,alpha,b1,b2);
					start += dim;
				}
			}
			std::vector<std::unique_ptr<Constraint>> Constraints;
		private:
			bool WasCopied = false;
			bool Impotent = false;
			ConstraintSet * DataOrigin;
			Vector c;

			void PotencyCheck()
			{
				if (Impotent)
				{
					std::cout << "\nCRITICAL ERROR\nYou are attempting to use a constraint whilst it is in an invalid state. \nConstraint object ownership is currently within another ConstraintSet, or within a Predictor. Please use the Retire() function to release the data." << std::endl;
					
					std::cout << "This error cannot be resolved: Terminating Program\n" << std::endl;
					exit(1);
				}
			}
	};


	inline ConstraintSet operator +(ConstraintSet & c1,  ConstraintSet & c2)
	{
		ConstraintSet out = c1;
		out.Add(c2);
		return out;
	}
	inline ConstraintSet operator +=(ConstraintSet & c1, ConstraintSet & c2)
	{
		c1.Add(c2);
		return c1;
	}
	template< class T, class S>
	inline ConstraintSet operator +(T & c1, S & c2)
	{
		ConstraintSet out = c1;

		out.Add(c2);
		return out;
	}
	
}