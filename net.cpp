#include "net.h"
#include "neuron.h"
#include "trainingSet.h"
//Для функции "assert()"
#include <cassert> 

double net::recentAverageSmoothingFactor = 100.0;

net::net(const vector<int> &topology)
{
	int numderLayers = topology.size();

	//Создание слоев и добавление в вектор layers(слоев)
	for (int layerNum = 0; layerNum < numderLayers; layerNum++)
	{
		layers.push_back(Layer());

		int numberOut = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		//Добавление нейронов в соответсвующий layer(слой)
		for (int neuronNum = 0; neuronNum <= topology[layerNum]; neuronNum++)
		{
			layers.back().push_back(neuron(numberOut, neuronNum));
		}
		layers.back().back().SetOutputVal(1.0);
	}
}


void net::getResults(vector<double> &resultVals) const
{
	resultVals.clear();

	for (unsigned n = 0; n < layers.back().size() - 1; ++n)
	{
		resultVals.push_back(layers.back()[n].getOutputVal());
	}
}

void net::backProp(const std::vector<double> &targetVals)
{

	Layer &outputLayer = layers.back();
	error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - outputLayer[n].getOutputVal();
		error += delta * delta;
	}
	error /= outputLayer.size() - 1;
	error = sqrt(error);


	recentAverageError =
		(recentAverageError * recentAverageSmoothingFactor + error)
		/ (recentAverageSmoothingFactor + 1.0);


	for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
	{
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	for (unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = layers[layerNum];
		Layer &nextLayer = layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}


	for (unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = layers[layerNum];
		Layer &prevLayer = layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}
}



void net::feedForward(const vector<double> &inputVals)
{
	vector<double> inputValsBack;
	//Исключения ошибки(вход должен быть равен количеству нейронов)
	assert(inputVals.size()==layers[0].size()-1);

	//Присвоение входных данных входным нейронам
	for (int i = 0; i < inputVals.size(); i++)
	{
		//Преобразование входных данных
			inputValsBack.push_back(2*tanh(-inputVals[i]));

			layers[0][i].SetOutputVal(inputValsBack[i]);
	}

	//Сложение выходов нейронов с предыдущего слоя и прогон через нелинейность
	for (int layerNum = 1; layerNum < layers.size(); layerNum++) 
	{
		Layer &prevLayer = layers[layerNum - 1];

		for (int n = 0; n < layers[layerNum].size()-1; n++)
		{
			layers[layerNum][n].FeedForward(prevLayer);
		}
	}

}

