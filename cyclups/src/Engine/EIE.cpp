#include "EIE.h"
// JSL::gnuplot gpu;
void cyclups::EIE::Recover(functionPointer curve, constraint::ConstraintSet constraint, kernel::Kernel K, basis::Basis B,int samples,int id, int seed)
{
	srand(seed);
	generator::randomiser.seed(seed);
	rand();//throw away first value
	id -=1;
	int res = samples;
	int scaleIdx = id / (noiseResolution * res);
	int noiseIdx = (id - scaleIdx *noiseResolution*res) / res;

	std::vector<double> ells = JSL::Vector::linspace(scaleMin,scaleMax,scaleResolution);
	std::vector<double> noises = JSL::Vector::linspace(noiseMin,noiseMax,noiseResolution);
	double scale = ells[scaleIdx];
	double noise = noises[noiseIdx];

	std::cout << scale << std::endl;
	K.UpdateParameter(0,scale);

	int N = rand() * 1.0/RAND_MAX *(maxData - minData) + minData;
	auto D = generator::NoisyXSample(N,curve,PredictLowerBound,PredictUpperBound,noise);

	std::vector<double> tt = JSL::Vector::linspace(PredictLowerBound,PredictUpperBound,PredictionResolution);
	auto Q = cyclups::Predictor(K,B,constraint);
	auto P = Q.Predict(tt,D,noise);
	JSL::gnuplot gpu;
	gpu.Scatter(D.X,D.Y);
	Curve(gpu,P.BLUP(),"BLUP",curve);
	Curve(gpu,P.CLUPS(),"CLUPS",curve);
	// gpu.Plot(P.BLUP().X,P.BLUP().Y,JSL::LineProperties::Colour("hold"));
	// gpu.Plot(P.CLUPS().X,P.CLUPS().Y);
	gpu.SetLegend(true);
	gpu.Show();
	double clupsError = TrueError(P.CLUPS(),curve);
	double blupError = TrueError(P.BLUP(),curve);
	std::cout << clupsError << " " << blupError << std::endl;
}

void cyclups::EIE::Plot(std::string file, std::string name)
{
	std::vector<double> ells = JSL::Vector::linspace(scaleMin,scaleMax,scaleResolution);
	std::vector<double> noises = JSL::Vector::linspace(noiseMin,noiseMax,noiseResolution);

	std::vector<std::vector<int>> NonCoincidentCounts(noiseResolution,std::vector<int>(scaleResolution,0));
	std::vector<std::vector<double>> ImprovementCounts(noiseResolution,std::vector<double>(scaleResolution,0));
	std::vector<std::vector<double>> DeprovementAmounts(noiseResolution,std::vector<double>(scaleResolution,0));
	std::vector<std::vector<double>> ImprovementAmounts(noiseResolution,std::vector<double>(scaleResolution,0));
	std::vector<std::vector<double>> CoindcidentCounts(noiseResolution,std::vector<double>(scaleResolution,0));
	int lIndex = 0;
	int nIndex = 0;
	bool first = true;
	double prevL = 0;
	double prevN = 0;
	int q = 0;
	double worst = 0;
	bool noneFound = true;
	forLineVectorIn(file,' ',
		++q;
		int lIndex = stoi(FILE_LINE_VECTOR[2]);
		int nIndex = stoi(FILE_LINE_VECTOR[3]);
		double scale = ells[lIndex];
		double noise = noises[nIndex];
		
		double coincidenceCheck = std::stod(FILE_LINE_VECTOR[8]);
		double diff = std::stod(FILE_LINE_VECTOR[7]);
		int N = std::stoi(FILE_LINE_VECTOR[4]);
		if (abs(coincidenceCheck) > 1e-4)
		{
			++NonCoincidentCounts[nIndex][lIndex];

			if (diff < 0)
			{
				ImprovementAmounts[nIndex][lIndex] += abs(diff);
				++ImprovementCounts[nIndex][lIndex];
			}
			else
			{
				DeprovementAmounts[nIndex][lIndex] += diff;
				noneFound = false;
				if (diff > worst)
				{
					worst = diff;
				}
			}
		}

	
	);
	int res = q/(scaleResolution * noiseResolution);

	double bestWorst = 0;

	double worstImprovement = -1;
	double bestImprovement = -1;
	for (int i = 0; i < noiseResolution; ++i)
	{
		for (int j  = 0; j < scaleResolution; ++j)
		{
			double orig = ImprovementAmounts[i][j];
			ImprovementAmounts[i][j] /= ImprovementCounts[i][j];
			CoindcidentCounts[i][j] = (NonCoincidentCounts[i][j])*100.0/res;
			if (ImprovementCounts[i][j] > 0)
			{

				double v = ImprovementAmounts[i][j];
				if (worstImprovement == -1)
				{
					worstImprovement = v;
					bestImprovement = v;
				}
				else
				{
					if (v < worstImprovement)
					{
						worstImprovement = v;
					}
					if (v > bestImprovement)
					{
						bestImprovement = v;
				}
			}

				int deprove = (NonCoincidentCounts[i][j] - ImprovementCounts[i][j]);
				DeprovementAmounts[i][j] /= deprove;
				ImprovementCounts[i][j] = ImprovementCounts[i][j] * 100.0/NonCoincidentCounts[i][j];

			}
		}
	}

	JSL::gnuplot gp;
	gp.WindowSize(1200,900);
	gp.SetMultiplot(2,2);
	gp.SetSuperTitle(name);
	gp.SetFontSize(JSL::Fonts::Global,10);
	gp.SetFontSize(JSL::Fonts::SuperTitle,20);
	gp.SetAxis(0);
	gp.Map(ells,noises,CoindcidentCounts);
	gp.SetXRange(scaleMin,scaleMax);
	gp.SetYRange(noiseMin,noiseMax);
	gp.SetXLabel("");
	gp.SetYLabel("Standard Error of Data");
	gp.SetCBLabel("\% Non-Coincident Models");

	gp.SetAxis(1);
	gp.Map(ells,noises,ImprovementCounts);
	gp.SetXRange(scaleMin,scaleMax);
	gp.SetYRange(noiseMin,noiseMax);
	gp.SetXLabel("");
	gp.SetYLabel("");
	gp.SetCBLabel("\% Non-Coincident with CLUPS Improvement");

	gp.SetAxis(2);
	gp.Map(ells,noises,ImprovementAmounts);
	gp.SetXRange(scaleMin,scaleMax);
	gp.SetYRange(noiseMin,noiseMax);
	gp.SetXLabel("");
	gp.SetCBLog(true);
	gp.SetColourRange(worstImprovement,bestImprovement);
	gp.SetCBLabel("Mean CLUPS Improvement");
	gp.SetYLabel("Standard Error of Data");
	gp.SetXLabel("Kernel Length Scale");
	gp.SetCBTicPowerFormat(true);
	gp.SetAxis(3);
	gp.Map(ells,noises,DeprovementAmounts);
	gp.SetXRange(scaleMin,scaleMax);
	gp.SetYRange(noiseMin,noiseMax);
	gp.SetCBTicPowerFormat(true);
	gp.SetCBLabel("Mean CLUPS Failure");
	gp.SetYLabel("");
	gp.SetXLabel("Kernel Length Scale");
	if (!noneFound)
	{
		gp.SetCBLog(true);
	}
	gp.Show();
}

void cyclups::EIE::InternalRun(functionPointer curve, constraint::ConstraintSet constraint, kernel::Kernel K, basis::Basis B, int Resolution, std::string outfile)
{
	std::vector<double> ells = JSL::Vector::linspace(scaleMin,scaleMax,scaleResolution);
	std::vector<double> noises = JSL::Vector::linspace(noiseMin,noiseMax,noiseResolution);

	// std::vector<std::vector<int>> NonCoincidentCounts(scaleResolution,std::vector<int>(noiseResolution,0));
	// std::vector<std::vector<int>> ImprovementCounts(scaleResolution,std::vector<int>(noiseResolution,0));
	// std::vector<std::vector<double>> ImprovementCounts(scaleResolution,std::vector<double>(noiseResolution,0));

	int seedSet = 1;
	int id = 1;
	std::vector<double> tt = JSL::Vector::linspace(PredictLowerBound,PredictUpperBound,PredictionResolution);

	JSL::initialiseFile(outfile);
	std::ostringstream out;

	JSL::ProgressBar<2,true,'|',50> PB(scaleResolution,noiseResolution);
	for (int scaleIdx = 0; scaleIdx < scaleResolution; ++scaleIdx)
	{
		double scale = ells[scaleIdx];
		K.UpdateParameter(0,scale);
		for (int noiseIdx = 0; noiseIdx < noiseResolution; ++noiseIdx)
		{
			double noise = noises[noiseIdx];

			for (int sample = 0; sample < Resolution; ++sample)
			{
				srand(seedSet); //ensures that we can exactly reproduce the curves, given we know the seed which generated the data! USeful for diving in to see what went wrong
				int r = rand();
				int N = rand() * 1.0/RAND_MAX *(maxData - minData) + minData;
				generator::randomiser.seed(seedSet);

				
				auto D = generator::NoisyXSample(N,curve,PredictLowerBound,PredictUpperBound,noise);

				auto Q = cyclups::Predictor(K,B,constraint);
				auto P = Q.Predict(tt,D,noise);
				

				double s = 0;
				for (int i = 0; i < PredictionResolution; ++i)
				{
					s += abs((P.Y[i]- P.Y_BLUP[i])/(1e-7 + P.Y[i]));
				}
				s/= PredictionResolution;

				double clupsError = TrueError(P.CLUPS(),curve);
				double blupError = TrueError(P.BLUP(),curve);
				out << id << " " << seedSet << " " << scaleIdx << " " << noiseIdx << " " << N << " " << clupsError << " " << blupError << " " << (clupsError - blupError) << " " << s <<"\n";
				seedSet = rand();
				++id;
				
			}
			PB.Update(scaleIdx,noiseIdx);
		}
	}
	JSL::writeStringToFile(outfile,out.str());
}

void cyclups::Curve(JSL::gnuplot &gp, cyclups::PairedData curve, std::string name, double (*func)(double))
{
	namespace lp = JSL::LineProperties;
	double err = cyclups::TrueError(curve,func); 
	std::ostringstream out;
    out.precision(3);
	out << err;
	std::string leg = name + " (ε = " + out.str() + ")";
	gp.Plot(curve.X,curve.Y,lp::Legend(leg),lp::PenSize(2));
}
