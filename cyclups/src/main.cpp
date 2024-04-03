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
		cyclups::Curve(gp,p.BLP(),"BLP",func);
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

int main(int argc, char**argv)
{
	JSL::Argument<int> Seed(time(NULL),"s",argc,argv);
	cyclups::generator::randomiser = std::default_random_engine(Seed);
	JSL::Argument<int> NSteps(1000,"n",argc,argv);




	auto K = cyclups::kernel::SquaredExponential(0.5,0.1);
	auto B = cyclups::basis::Hermite(2);
	auto C = cyclups::constraint::Positive();

	// auto E = cyclups::EIE();

	// int R = 100;
	// E.Run(testFunc,C,K,B,R,"test.tst");
	// E.Plot("test.tst","Test 1.0");
	// E.Recover(testFunc,C,K,B,R,4343,1397452809 );

	// auto C2 = cyclups::constraint::Integrable(0.999937);
	// E.Run(testFunc,C2,K,B,R,"test.tst");
	// E.Plot("test.tst","Test 0.999937");
	// srand(Seed);

	//generate sample
	double xmin = -4;
	double xmax = 4;
	int res = 21;
	double dataError = 0.04;
	auto D = cyclups::generator::NoisyXSample(res,testFunc,xmin,xmax,dataError);

	//define predictor
	// auto c2 = cyclups::constraint::Integrable(1);
	// auto combined = c2;
	

	auto P = cyclups::Predictor(K,B,C);
	P.Optimiser.MaxSteps = NSteps;
	//make predictions
	std::vector<double> tt = JSL::Vector::linspace(xmin,xmax,331);
	// auto p = P.Predict(tt,D,0.1);

	JSL::gnuplot gp;
	PlotBackground(gp,D,testFunc);
	auto p = P.Predict(tt,D,dataError);
	Plot(gp,p,testFunc,"CLUPS");
	
	// P.Retire();
	std::vector<int> oms = {10,50,100};
	for (auto i : oms)
	{
		omega = i;
		// auto P2 = cyclups::Predictor(K,B,C);
		// P2.Optimiser.MaxSteps = NSteps;
		auto pp = P.RegularisedPrediction(tt,D,dataError,cyclups::RegularisingFunction(R,gradR));

	
		Plot(gp,pp,testFunc,"rCLUPS_{" + std::to_string(omega) + "}");
	}
	gp.Show();
}