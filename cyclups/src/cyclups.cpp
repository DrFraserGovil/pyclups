#include "JSL.h"
#include "dataArrays.h"
#include "generator.h"
#include <Dense>
#include "kernel.h"
#include "basis.h"
#include "predictor.h"
#include "constraintVector.h"
#include <sstream>
#include "my_constraints.h"
double testFunc(double x)
{
	// return 1.0/(1 + exp(-x));
	return 1.0/sqrt(2*M_PI) * exp( - x*x/2)-0.1;
}

void Curve(JSL::gnuplot & gp, cyclups::PairedData curve, std::string name)
{
	namespace lp = JSL::LineProperties;
	double err = cyclups::TrueError(curve,testFunc); 
	std::ostringstream out;
    out.precision(3);
	out << err;
	std::string leg = name + " (ε = " + out.str() + ")";
	gp.Plot(curve.X,curve.Y,lp::Legend(leg));
}
void Plot(cyclups::functionPointer trueFunc, cyclups::PairedData data, cyclups::Prediction p)
{
	
	int plotRes = 100;
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

	Curve(gp,p.BLP(),"BLP");
	Curve(gp,p.BLUP(),"BLUP");
	Curve(gp,p.CLUPS(),"CLUPS");
	
	gp.SetLegend(true);
	gp.Show();
}





int main(int argc, char**argv)
{
	JSL::Argument<int> Seed(time(NULL),"s",argc,argv);
	cyclups::generator::randomiser = std::default_random_engine(Seed);
	srand(Seed);



	// auto r = cyclups::constraint::Integrable(2);
	auto c1 = cyclups::constraint::LessThan([](double t){return 0.2;},[](double t){return t > 0;});
	auto c2 = cyclups::constraint::GreaterThan([](double t){return -0.025*t;},[](double t){return t < 0;});
	auto combined = c2 + c1;
	double xmin = -3;
	double xmax = 3;
	int res = 21;
	auto D = cyclups::generator::NoisyXSample(res,testFunc,xmin,xmax,0.05);
	// for (int i = 0; i < 2; ++i)
	// {
		// cyclups::constraint::ConstraintSet q(r);


		std::vector<double> tt = JSL::Vector::linspace(xmin,xmax,121);
		auto K = cyclups::kernel::SquaredExponential(0.1,1);
		auto B = cyclups::basis::Hermite(3);
		
		auto P = cyclups::Predictor(K,B,combined);
		auto p = P.Predict(tt,D,0.1);
		Plot(testFunc,D,p);
	// }

	std::cout << "Ending program" << std::endl;
}