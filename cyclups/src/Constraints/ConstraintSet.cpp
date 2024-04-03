#include "ConstraintSet.h"

namespace cyclups
{
cyclups::constraint::ConstraintSet::ConstraintSet()
{
	Dimension = 0;
}

cyclups::constraint::ConstraintSet::ConstraintSet(ConstraintSet &copy)
{
	copy.PotencyCheck();
	PotencyCheck();
	Constraints = std::move(copy.Constraints);
	WasCopyConstructed = true;
	copy.Impotent = true;
	DataOrigin = &copy;
	
}

cyclups::constraint::ConstraintSet::~ConstraintSet()
{
	Retire();
}

void cyclups::constraint::ConstraintSet::Add(ConstraintSet c)
{
	PotencyCheck();
	c.PotencyCheck();
	Constraints = std::move(c.Constraints);
}

void constraint::ConstraintSet::Remove()
{
	Constraints.pop_back();
}

void cyclups::constraint::ConstraintSet::Transfer(ConstraintSet *c)
{
	c->Constraints = std::move(Constraints);
}

void cyclups::constraint::ConstraintSet::PotencyCheck()
{
	if (Impotent)
	{
		std::cout << "\nCRITICAL ERROR\nYou are attempting to use a constraint whilst it is in an invalid state. \nConstraint object ownership is currently within another ConstraintSet, or within a Predictor. Please use the Retire() function to release the data." << std::endl;
		
		std::cout << "This error cannot be resolved: Terminating Program\n" << std::endl;
		exit(1);
	}
}

void cyclups::constraint::ConstraintSet::Retire()
{
	if (WasCopyConstructed)
	{
		Transfer(DataOrigin);
		DataOrigin->Impotent = false;
	}
	Impotent = true;
}

void cyclups::constraint::ConstraintSet::SetPosition(Vector c)
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

void cyclups::constraint::ConstraintSet::Initialise(const std::vector<double> &t)
{
	for (int i = 0; i < Constraints.size(); ++i)
	{
		Constraints[i]->CallInitialiser(t);
	}
	//can't add matrices together until size determined
	Dimension = 0;
	TransformDimension = 0;
	for (int i =0; i < Constraints.size(); ++i)
	{
		Dimension += Constraints[i]->Dimension;
		TransformDimension += Constraints[i]->TransformDimension;
	}
	IsConstant = (TransformDimension == 0);
	B = Matrix::Zero(Dimension,t.size());
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

cyclups::Vector & cyclups::constraint::ConstraintSet::C()
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

cyclups::Matrix &cyclups::constraint::ConstraintSet::Gradient()
{
	int c_Ind = 0;
	int w_Ind = 0;
	for (int i = 0; i < Constraints.size(); ++i)
	{
		int dim = Constraints[i]->Dimension;
		int wDim = Constraints[i]->TransformDimension;
		if (wDim > 0)
		{
			auto & r = Constraints[i]->vector.Gradient();
			Grad.block(w_Ind,c_Ind,wDim,dim) << r;
		}
		c_Ind += dim;
		w_Ind += wDim;
	}
	return Grad;
}

void cyclups::constraint::ConstraintSet::Step(Vector &grad, int l, const OptimiserProperties &op)
{
	int start = 0;
	for (int i = 0; i < Constraints.size(); ++i)
	{
		int dim = Constraints[i]->TransformDimension;
		auto v = VectorSlice(grad,start,dim);
		Constraints[i]->vector.Step(v,l,op);
		start += dim;
	}
}

void cyclups::constraint::ConstraintSet::SavePosition()
{
	for (int i = 0; i < Constraints.size(); ++i)
	{
		Constraints[i]->vector.SavePosition();
	}
}

void cyclups::constraint::ConstraintSet::RecoverPosition()
{
	for (int i = 0; i < Constraints.size(); ++i)
	{
		Constraints[i]->vector.RecoverPosition();
	}
}

}