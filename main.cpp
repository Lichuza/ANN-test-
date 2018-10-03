#include "trainingSet.h"
#include "neuron.h"
#include "net.h"

void showVectorVals(std::string label, std::vector<double> &v)
{
	std::cout << label << " ";
	for (int i = 0; i < v.size(); i++)
	{
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}

void display(std::string label, std::vector<double> &v)
{
	ofstream out("C:/result.txt", ios::app);
	for (int i = 0; i < v.size(); i++)
	{	
		out << label << v[i]<<" ";
	}
	out << endl;
	out.close();	
}

int main()
{
	trainingSet trainingData("C:/DataSet.txt");
	vector<int> topology;
	trainingData.getTopology(topology);
	net net(topology);

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	//Обучение ИНС на 70% от исходных данных
	while (!trainingData.isEOF())
	{
		++trainingPass;
		std::cout << std::endl << "Pass: " << trainingPass << std::endl;

		if (trainingData.getNextInputs(inputVals) != topology[0])
			break;
		showVectorVals("Input:", inputVals);
		net.feedForward(inputVals);

		trainingData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		net.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		net.backProp(targetVals);

		std::cout << "Net average error: " << net.getRecentAverageError() << std::endl;
	}

	//Тестирование ИНС на 30% от исходных данных
	trainingSet testData("C:/testData.txt");
	int testPass = 0, b=0;
	testData.getTopology(topology);

	cout << endl << "Testing process" << endl;

	while (!testData.isEOF())
	{
		++testPass;

		if (testData.getNextInputs(inputVals) != topology[0])
			break;

		net.feedForward(inputVals);

		ofstream out("C:/result.txt", ios::app);
		out << "Pass: " << b << endl;
		b++;

		testData.getTargetOutputs(targetVals);
		display("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		net.getResults(resultVals);
		display("Outputs:", resultVals);

		out << endl;
		out.close();	
	}

	cout << endl << "Pocess is completed" <<endl;
	system("PAUSE");
	return(0);
}