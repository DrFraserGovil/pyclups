#define GNUPLOT_NO_TIDY
#include "JSL.h"
#include <sstream>

#include "cyclups.h"


void Plot(cyclups::functionPointer trueFunc, cyclups::PairedData data, cyclups::Prediction p, double (*func)(double))
{
	
	int plotRes = 1000;
	std::vector<double> realX = JSL::Vector::linspace(data.X[0],data.X[data.X.size() -1],plotRes);
	std::vector<double> realY(plotRes);
	for (int i = 0; i < plotRes; ++i)
	{
		realY[i] = trueFunc(realX[i]);
	}
	namespace lp = JSL::LineProperties; 
	JSL::gnuplot gp;
	gp.Plot(realX,realY,lp::Legend("True Function"));
	gp.Scatter(data.X,data.Y,lp::Legend("Sampled Data"));

	cyclups::Curve(gp,p.BLP(),"BLP",func);
	cyclups::Curve(gp,p.BLUP(),"BLUP",func);
	cyclups::Curve(gp,p.CLUPS(),"CLUPS",func);
	
	gp.SetLegend(true);
	gp.Show();
}


double testFunc(double x)
{
	// x = x +0.6;
	// return 1.0/(1 + exp(-x));
	return 1.0/sqrt(2*M_PI) * exp( - x*x/2);
}

int main(int argc, char**argv)
{
	JSL::Argument<int> Seed(time(NULL),"s",argc,argv);
	cyclups::generator::randomiser = std::default_random_engine(Seed);
	JSL::Argument<int> NSteps(1000,"n",argc,argv);



	auto K = cyclups::kernel::SquaredExponential(0.3,0.1);
	auto B = cyclups::basis::Hermite(3);
	auto C = cyclups::constraint::PositiveUnimodal();

	auto E = cyclups::EIE();

	// int R = 100;
	// E.Run(testFunc,C,K,B,R,"test.tst");
	// E.Plot("test.tst","Test 1.0");
	// E.Recover(testFunc,C,K,B,R,4343,1397452809 );

	// auto C2 = cyclups::constraint::Integrable(0.999937);
	// E.Run(testFunc,C2,K,B,R,"test.tst");
	// E.Plot("test.tst","Test 0.999937");
	// srand(Seed);

	//generate sample
	double xmin = -3;
	double xmax = 3;
	int res = 21;
	auto D = cyclups::generator::NoisyXSample(res,testFunc,xmin,xmax,0.05);

	//define predictor
	// auto c2 = cyclups::constraint::Integrable(1);
	// auto combined = c2;
	
	auto P = cyclups::Predictor(K,B,C);
	P.Optimiser.MaxSteps = NSteps;
	//make predictions
	std::vector<double> tt = JSL::Vector::linspace(xmin,xmax,131);
	auto p = P.Predict(tt,D,0.1);

	Plot(testFunc,D,p,testFunc);
}