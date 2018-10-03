#pragma once
#include "Neuron.h"
#include<vector>
using namespace std;

class net
{
public:
	//В конструкторе передача топологии сети, количество нейронов
	net(const vector<int> &topology);

	//Наполнение сети данными
	void feedForward(const vector<double> &inputVals);

	//Метод тренировки сети
	void backProp(const vector<double> &targetVals);

    //Результат
	void getResults(vector<double> &ResultsVals) const;

	//Подсчет ошибки
	double getRecentAverageError() const { return recentAverageError; }

private:
	//Форма обращения к вектору Layer
	std::vector<Layer> layers;

	double error;
	double recentAverageError;
	static double recentAverageSmoothingFactor;
};

