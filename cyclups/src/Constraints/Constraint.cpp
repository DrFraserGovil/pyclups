#include "Constraint.h"

cyclups::constraint::InitialiseContainer::InitialiseContainer()
{
	Initialised = false;
}

cyclups::constraint::InitialiseContainer::InitialiseContainer(ConstraintVector &v, Matrix &m)
{
	vector = v;
	matrix = m;
	Initialised = true;
}

cyclups::constraint::Constraint::Constraint()
{
}

cyclups::constraint::Constraint::Constraint(Matrix mat, ConstraintVector vec)
{
	Dimension = vec.Dimension;
	IsConstant = true;
	TransformDimension = vec.TransformDimension;
	vector = vec;
	matrix = mat;
}
void cyclups::constraint::Constraint::CallInitialiser(const std::vector<double> &ts)
{
	InitialiseContainer init = Initialiser(ts);
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

cyclups::constraint::InitialiseContainer cyclups::constraint::Constraint::Initialiser(const std::vector<double> &t)
{
	return InitialiseContainer();
}

std::vector<double> cyclups::constraint::Constraint::ApplyDomain(const std::vector<double> &t)
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
