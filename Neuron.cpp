#include "Neuron.h"
#include "iostream"

double neuron::eta = 0.15;
double neuron::alpha = 0.5;

//Связи нейронов
neuron::neuron(int numberOut, int myIndex)
{
	for (int i = 0; i < numberOut; i++)
	{
		outputWeights.push_back(Connection());
		outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}

void neuron::FeedForward(const Layer &prevLayer)
{
	double sum = 0.0;

	for (int n=0;n<prevLayer.size();n++) 
	{
		sum += prevLayer[n].getOutputVal()*prevLayer[n].outputWeights[m_myIndex].weight;
	}
	outputVal = neuron::activationFunction(sum);
}


void neuron::updateInputWeights(Layer &prevLayer)
{
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		neuron &neuron = prevLayer[n];
		double oldDeltaWeight = neuron.outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * gradient + alpha * oldDeltaWeight;
		neuron.outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double neuron::sumDOW(const Layer &nextLayer) const
{
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += outputWeights[n].weight * nextLayer[n].gradient;
	}

	return sum;
}

void neuron::calcHiddenGradients(const Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	gradient = dow * neuron::activationDerivateFunction(outputVal);
}


void neuron::calcOutputGradients(double targetVals)
{
	double delta = targetVals - outputVal;
	gradient = delta * neuron::activationDerivateFunction(outputVal);
}


double neuron::activationFunction(double x)
{
	//Сигмоида
	return 1 / (1 + exp(-x));
}

double neuron::activationDerivateFunction(double x) 
{
	//Производная сигмоиды
	return (1 / (1 + exp(-x)))*(1 - (1 / (1 + exp(-x))));
}

