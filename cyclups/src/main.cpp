// #define GNUPLOT_NO_TIDY
#include "JSL.h"
#include <sstream>

#include "cyclups.h"

void PlotBackground(JSL::gnuplot & gp, cyclups::PairedData data,double (*trueFunc)(double))
{
	int plotRes = 1000;
	std::vector<double> realX = JSL::Vector::linspace(data.X[0],data.X[data.X.size() -1],plotRes);
	std::vector<double> realY(plotRes);
	for (int i = 0; i < plotRes; ++i)
	{
		realY[i] = trueFunc(realX[i]);
	}
	namespace lp = JSL::LineProperties; 
	gp.Plot(realX,realY,lp::Legend("True Function"));
	gp.Scatter(data.X,data.Y,lp::Legend("Sampled Data")),lp::Colour("hold");
	gp.SetLegend(true);
}
bool firstPlot = true;
void Plot(JSL::gnuplot & gp, cyclups::Prediction p, double (*func)(double),std::string name)
{
	namespace lp = JSL::LineProperties; 
	if (firstPlot)
	{
		// cyclups::Curve(gp,p.BLP(),"BLP",func);
		cyclups::Curve(gp,p.BLUP(),"BLUP",func);
		firstPlot = false;
	}
	cyclups::Curve(gp,p.CLUPS(),name,func);
	// gp.SetYLog(true);
}


double testFunc(double x)
{
	// x = x +0.6;
	// return 1.0/(1 + exp(-x));
	return 1.0/sqrt(2*M_PI) * exp( - x*x/2);
}

double omega = 10;
double R(const cyclups::Vector & in)
{
	double s = 0;
	for (int i = 1; i < in.size(); ++i)
	{
		double v = in[i] - in[i-1];
		s += v*v;
	}
	return omega*s;
}
double R_curve(const cyclups::Vector & in)
{
	double s = 0;
	for (int i = 2; i < in.size(); ++i)
	{
		double v = in[i] - 2*in[i-1] + in[i-2];
		s += v*v;
	}
	return omega*s;
}
void gradR_curve(cyclups::Vector & out, const cyclups::Vector & in)
{
	int N = in.size();
	for (int i =0; i < N; ++i)
	{
		double v = 0;
		if (i == 0)
		{
			v = in[0] -2 * in[1] + in[2];	
		}
		else if (i == 1)
		{
			v = 5*in[1] - 2 * in[0] - 4 * in[2] + in[3];
		}
		else if (i ==N-1)
		{
			v = in[i-2] + -2 * in[i-1] + in[i];
		}
		else if (i == N-2)
		{
			v = in[i-2] -4*in[i-1] + 5*in[i] - 2 * in[i+1];
		}
		else
		{
			v = in[i-2] - 4 * in[i-1] + 6*in[i] - 4 * in[i+1] + in[i+2];
		}
		out[i] = 2*omega*v;
	}
}


void gradR(cyclups::Vector & out, const cyclups::Vector & in)
{
	// out = cyclups::Vector::Zero(out.size());
	for (int i = 0; i < in.size(); ++i)
	{
		double v = 0;
		if (i > 0)
		{
			v += 2*(in[i] - in[i-1]);
		}
		if (i < in.size() -1)
		{
			v += 2*(in[i] - in[i+1]);
		}
		out[i] = v * omega;
	}
	

}

bool excluder(double x)
{
	return x < 0;
}

int main(int argc, char**argv)
{
	JSL::Argument<int> Seed(time(NULL),"s",argc,argv);
	cyclups::generator::randomiser = std::default_random_engine(Seed);
	JSL::Argument<int> NSteps(1000,"n",argc,argv);



	//initialise
	auto K = cyclups::kernel::SquaredExponential(0.7,0.1);
	auto B = cyclups::basis::Hermite(2);
	auto C = cyclups::constraint::Positive();
	//generate sample
	double xmin = -4;
	double xmax = 4;
	int res = 15;
	double dataError = 0.02;
	auto D = cyclups::generator::UniformXSample(res,testFunc,xmin,xmax,dataError);

	//set up predictor
	auto P = cyclups::Predictor(K,B,C);
	P.Optimiser.MaxSteps = NSteps;
	
	//make predictions
	std::vector<double> tt = JSL::Vector::linspace(xmin,xmax,111);
	auto p = P.Predict(tt,D,dataError);

	//plot
	JSL::gnuplot gp;
	gp.WindowSize(1500,800);
	gp.SetMultiplot(1,2);
	gp.SetAxis(0);
	gp.SetTitle("Smooth Regulariser");
	PlotBackground(gp,D,testFunc);
	Plot(gp,p,testFunc,"CLUPS");
	firstPlot = true;
	gp.SetAxis(1);
	gp.SetTitle("Curvature Regulariser");
	PlotBackground(gp,D,testFunc);
	Plot(gp,p,testFunc,"CLUPS");

	std::vector<double> testOmega = {10,100,1000};
	for (auto i : testOmega)
	{
		omega = i;
		auto pp = P.RegularisedPrediction(tt,D,dataError,cyclups::RegularisingFunction(R_curve, gradR_curve));

		omega = i/10;
		auto pp2 = P.RegularisedPrediction(tt,D,dataError,cyclups::RegularisingFunction(R, gradR));
		std::ostringstream ss1;
		ss1 << std::fixed << std::setprecision(2) << "rCLUPS ("  <<  omega*10 << ")";
		std::ostringstream ss2;
		ss2 << std::fixed << std::setprecision(2) << "rCLUPS ("  <<  omega << ")";
		gp.SetAxis(1);
		Plot(gp,pp,testFunc,ss1.str());
		gp.SetAxis(0);
		Plot(gp,pp2,testFunc,ss2.str());
	}
	
	gp.Show();
}