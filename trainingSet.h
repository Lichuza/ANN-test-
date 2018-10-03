#pragma once
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
using namespace std;
class trainingSet
{
public:
	trainingSet(const string filename);
	bool isEOF() { return trainingDataFile.eof(); }
	void getTopology(vector<int> &topology);
	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
	ifstream trainingDataFile;
};
