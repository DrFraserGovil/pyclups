#include "predictor.h"



namespace cyclups
{
	// Predictor::Predictor(kernel::Kernel k, basis::Basis b, constraint::ConstraintSet &constraint): Kernel(k), Basis(b), Constraint(constraint){}


	

	void Predictor::Initialise(cvec PredictX, const PairedData &data, cvec dataErrors)
	{
		// for (int i = 0; i < Cons)
		Constraint.Initialise(PredictX);
		int nData = data.X.size();
		int nPredict = PredictX.size();

		//stores for dot product
		Vector x = Vector::Map(&data.Y[0],data.Y.size());


		//store basis matrix
		Matrix Phi = Matrix(Basis.MaxOrder+1,nData);
		for (int b = 0; b <=Basis.MaxOrder; ++b)
		{
			for (int i = 0; i < nData; ++i)
			{
				Phi(b,i) = Basis(b,data.X[i]);
			}
		}
		Store.p_blps = std::vector<double>(nPredict);	
		Store.p_blups = std::vector<double>(nPredict);		
		Store.a_blups.resize(nPredict);
		Store.ks.resize(nPredict);
		Matrix K = Kernel.GetMatrix(data.X,dataErrors);
		Store.K = K;
		Matrix Phi_T = Phi.transpose();
		auto K_decomp = K.ldlt();
		auto KinvPhiT = K_decomp.solve(Phi.transpose());
		Matrix C = Phi * KinvPhiT;
		auto C_decomp = C.ldlt();
		for (int i = 0; i < PredictX.size(); ++i)
		{
			Vector ki = Kernel.GetVector(PredictX[i],data.X,dataErrors);
			Vector phi = Basis.GetVector(PredictX[i]);
			Vector a_blp = K_decomp.solve(ki);
			Vector blup_correct = K_decomp.solve(Phi_T * C_decomp.solve(Phi * a_blp));
			Vector blup_fit = K_decomp.solve(Phi_T * C_decomp.solve(phi));  
			
			Vector a_blup = a_blp - blup_correct + blup_fit;
			Store.a_blups[i] = a_blup;
			Store.p_blps[i] = x.dot(a_blp);
			Store.p_blups[i] = x.dot(a_blup);
			Store.ks[i] = ki;
		}

		//used for computing the score later
		// auto r = Phi;
		// auto t = C_decomp.solve(Phi);

		Matrix corr = K_decomp.solve(Phi.transpose() * C_decomp.solve(Phi));
		Matrix DeltaMatrix = Matrix::Identity(nData,nData) - corr.transpose(); 

		Vector Dx = DeltaMatrix *x;
		Store.Delta = K_decomp.solve(Dx);
		Store.Beta = x.dot(Store.Delta);
	}
	Prediction Predictor::Predict(cvec predictX,const PairedData & data, double dataErrors)
	{
		std::vector<double> eVec(data.X.size(),dataErrors);
		return Predict(predictX, data,eVec);
	}
	Prediction Predictor::Predict(cvec predictX,const PairedData & data, cvec dataErrors)
	{
		Initialise(predictX,data,dataErrors);

		auto B = Constraint.B;
		auto pblup = Vector::Map(&Store.p_blups[0],predictX.size());
		Store.Bp_blups = B * pblup;
		Store.BBt = (B * B.transpose()).ldlt();

		if (!Constraint.IsConstant)
		{
			Optimise(predictX,data,dataErrors);
		}


		auto c = Constraint.C();
		auto BzminusC = Store.Bp_blups - c;
		auto corrector  = B.transpose() * Store.BBt.solve(BzminusC);

		std::vector<double> p_clups(predictX.size(),0);
		for (int i = 0; i < p_clups.size(); ++i)
		{
			p_clups[i] = Store.p_blups[i] - corrector[i];
		}
		return Prediction(predictX,p_clups,Store.p_blups,Store.p_blps);
	}


	double Predictor::ComputeScore(cvec predictX)
	{
		Vector BzminusC = Store.Bp_blups - Constraint.C();
		Vector corrector  = Constraint.B.transpose() * Store.BBt.solve(BzminusC);

		
		double score = 0;
		for (int i =0; i < predictX.size(); ++i)
		{
			auto ai = Store.a_blups[i] + corrector[i]/Store.Beta * Store.Delta;
			score += Kernel(predictX[i],predictX[i]) +  ai.dot(Store.K * ai) - 2 * Store.ks[i].dot(ai);
		}
		return score;
	}

	void Predictor::Optimise(cvec predictX, const PairedData & data, cvec dataErrors)
	{
		Constraint.SetPosition(Store.Bp_blups);
		Optimiser.Clear();

		double bestScore = ComputeScore(predictX);
		Constraint.SavePosition();

		int l = 0;
		if (Optimiser.MaxSteps > 0)
		{
			while (!Optimiser.Converged)
			{
				Vector dLdc = Store.BBt.solve(Constraint.C() - Store.Bp_blups);
				Matrix dcdw = Constraint.Gradient();
				Vector dLdw = dcdw * dLdc;

				Constraint.Step(dLdw,l,Optimiser);

				if (l % 1 == 0)
				{
					double mse = ComputeScore(predictX);
					if (mse < bestScore)
					{
						bestScore = mse;
						Constraint.SavePosition();
					}
					
					Optimiser.CheckConvergence(l,dLdw.norm(),mse);
				}
				else
				{
					Optimiser.CheckConvergence(l,dLdw.norm());
				}
				

				++l;
			}
		}
		Constraint.RecoverPosition();
		Optimiser.PrintReason();
	}

	void Predictor::Retire()
	{
		Constraint.Retire();
	}

}