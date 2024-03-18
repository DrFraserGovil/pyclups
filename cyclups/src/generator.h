#pragma once
#include "dataArrays.h"
#include "JSL.h"
#include <random>
namespace cyclups
{


	namespace generator
	{
		extern std::default_random_engine randomiser;

		PairedData Sample(int N, functionPointer f, double xMin, double xMax, double noise);
		
		PairedData UniformXSample(int N, functionPointer f, double xMin, double xMax, double noise);

		PairedData NoisyXSample(int N, functionPointer f, double xMin, double xMax, double noise);

		PairedData RandomXSample(int N, functionPointer f, double xMin, double xMax, double noise);

	}
}