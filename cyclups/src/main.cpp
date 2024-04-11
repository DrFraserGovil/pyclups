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
	// if (firstPlot)
	// {
	// 	cyclups::Curve(gp,p.BLP(),"BLP",func);
	// 	cyclups::Curve(gp,p.BLUP(),"BLUP",func);
	// 	firstPlot = false;
	// }
	cyclups::Curve(gp,p.CLUPS(),name,func);
	// gp.SetYLog(true);
}


double testFunc(double x)
{
	// x = x +0.6;
	return 1.0/(1 + exp(x));
	// return 1.0/sqrt(2*M_PI) * exp( - x*x/2);
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
double R_grad(const cyclups::Vector & in)
{
	double s = 0;
	for (int i = 1; i < in.size(); ++i)
	{
		double v = in[i] - in[i-1];
		s += v*v;
	}
	return omega*s;
}
void gradR(cyclups::Vector & out, const cyclups::Vector & in)
{
	// out = cyclups::Vector::Zero(out.size());
	for (int i = 0; i < in.size(); ++i)
	{
		if (i > 0)
		{
			out[i] += 2*omega*(in[i] - in[i-1]);
		}
		if (i < in.size() -1)
		{
			out[i] += 2*omega*(in[i] - in[i+1]);
		}
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
	auto K = cyclups::kernel::SquaredExponential(1,0.1);
	auto B = cyclups::basis::Hermite(2);
	auto C = cyclups::constraint::Monotonic(cyclups::constraint::direction::Negative);

	//generate sample
	double xmin = -10;
	double xmax = 10;
	int res = 15;
	double dataError = 0.04;
	auto D = cyclups::generator::UniformXSample(res,testFunc,xmin,xmax,dataError);

	//set up predictor
	auto P = cyclups::Predictor(K,B,C);
	P.Optimiser.MaxSteps = NSteps;
	
	//make predictions
	std::vector<double> tt = JSL::Vector::linspace(xmin,xmax,111);
	auto p = P.Predict(tt,D,dataError);
	omega = 9;
	auto pp = P.RegularisedPrediction(tt,D,dataError,cyclups::RegularisingFunction(R,gradR));

	//plot
	JSL::gnuplot gp;
	PlotBackground(gp,D,testFunc);
	Plot(gp,p,testFunc,"CLUPS");
	Plot(gp,pp,testFunc,"rCLUPS");
	gp.Show();
}