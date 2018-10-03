#pragma once
#include <vector>
#include<math.h>
class neuron;

//������ �� ��������(����)
typedef std::vector<neuron> Layer;

//���� � ����������
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

	//"���� �������"-������������ � ������ ����� ������������
	void FeedForward(const Layer &prevLayer);

	//������� ��������� ��������� ����
	void calcOutputGradients(double targetVals);

	//������� ��������� �������� ����
	void calcHiddenGradients(const Layer &nextLayer);

	//���������� �����
	void updateInputWeights(Layer &prevLayer);
private:
	//������� ������� ��� ���������� ����� ���������� ����������
	static double randomWeight() { return rand() / double(RAND_MAX);}
	//����� �������
	double outputVal;
	//������ �����
	std::vector<Connection> outputWeights;

	//���������� �������
	double activationFunction(double x);
	double activationDerivateFunction(double x);
	double sumDOW(const Layer &nextLayer) const;

	//���������
	static double eta;
	static double alpha;

	int m_myIndex;
	double gradient;


};

