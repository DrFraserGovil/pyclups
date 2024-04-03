#include "OptimiserProperties.h"

void cyclups::OptimiserProperties::Clear()
{
	PrevScore = 0;
	GradientMemory= 0;
	ScoreMemory = 0;
	MaxAlpha = alpha*10;
	MinAlpha = alpha/10;
	TriggeringStep = 0;
	NegativeCounter = 0;
}
void cyclups::OptimiserProperties::CheckConvergence(int l, double gradnorm)
{
				
	if (l >= MaxSteps)
	{
		ReachedMaxSteps = true;
	}
	
	double earlyCorrector = 1.0/(1.0 - pow(ConvergenceMemory,l+1));
	
	GradientMemory = earlyCorrector * (ConvergenceMemory * GradientMemory + (1.0 - ConvergenceMemory) * gradnorm);

	if (GradientMemory < ConvergedGradient)
	{
		triggeringGradient = gradnorm;
		GradientConverged = true;
	}
	
	Converged = (ReachedMaxSteps || GradientConverged || ScoreConverged) && l > MinSteps;
	if (Converged)
	{
		TriggeringStep = l;
		alpha = MaxAlpha /10; //ensures it always goes back into the correct state
	}		
}

void cyclups::OptimiserProperties::CheckConvergence(int l, double gradnorm, double score)
{
	//do score bit
	
	if (l > 0)
	{
		double alphaCorrector = (10*alpha/MaxAlpha);
		double earlyCorrector = 1.0/(1.0 - pow(ConvergenceMemory,l+1));
		double scoreDelta = abs((score - PrevScore)/PrevScore);
		ScoreMemory = earlyCorrector * alphaCorrector * (ConvergenceMemory * ScoreMemory + (1.0 - ConvergenceMemory) * scoreDelta);
		if (ScoreMemory < ConvergedScore)
		{
			triggeringScore = scoreDelta;
			ScoreConverged = true;
		}
	}
	UpdateAlpha(score);
	PrevScore = score;
	CheckConvergence(l,gradnorm);
}

void cyclups::OptimiserProperties::PrintReason()
{
	std::cout << "The Optimiser halted at step " << TriggeringStep << " because:\n";
	if (ReachedMaxSteps)
	{
		std::cout << "\t-Reached max iteration count.\n";
	}
	if (GradientConverged)
	{
		std::cout << "\t-Mean-Gradient converged below " << ConvergedGradient << "(" << triggeringGradient << ")\n";
	}
	if (ScoreConverged)
	{
		std::cout << "\t-Mean-Score has not changed by more than " << 100*ConvergedScore << "% (" <<  triggeringScore << ")\n";
	}
}

void cyclups::OptimiserProperties::UpdateAlpha(double score)
{			
	if (score > PrevScore)
	{
		NegativeCounter +=2;
		if (NegativeCounter >= 10)
		{
			NegativeCounter = 0;
			alpha = std::max(MinAlpha,alpha *0.8);
		}
	}
	else
	{
		NegativeCounter -=1;
		if (NegativeCounter == -5)
		{
			NegativeCounter = 0;
			alpha = std::min(MaxAlpha,alpha * 1.02);
		}
	}
};
