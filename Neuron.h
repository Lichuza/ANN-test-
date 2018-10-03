#pragma once
#include <vector>
#include<math.h>
class neuron;

//Вектор из нейронов(слой)
typedef std::vector<neuron> Layer;

//Веса и дельтавеса
struct Connection
{
	double weight;
	double deltaWeight;

};

class neuron
{
public:
	neuron(int numberOut, int myIndex);

	void SetOutputVal(double val) { outputVal = val; }
	double getOutputVal() const { return outputVal; }

	//"Ядро нейрона"-суммирование и прогон через нелинейность
	void FeedForward(const Layer &prevLayer);

	//Подсчет градиента выходного слоя
	void calcOutputGradients(double targetVals);

	//Подсчет градиента скрытого слоя
	void calcHiddenGradients(const Layer &nextLayer);

	//Обнавление весов
	void updateInputWeights(Layer &prevLayer);
private:
	//Функция рандома для заполнения весов начальными значениями
	static double randomWeight() { return rand() / double(RAND_MAX);}
	//Выход нейрона
	double outputVal;
	//Вектор весов
	std::vector<Connection> outputWeights;

	//Нелинейная функция
	double activationFunction(double x);
	double activationDerivateFunction(double x);
	double sumDOW(const Layer &nextLayer) const;

	//Константы
	static double eta;
	static double alpha;

	int m_myIndex;
	double gradient;


};

