#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H


#include <vector>
#include "vectoroperations.h"
#include <math.h>
#include <random>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>


class NeuralNetwork
{
public:
    NeuralNetwork(int layerCount, int neuronCount, int targetCount, int inputCount);

    ~NeuralNetwork();

    std::vector<double> getRandomDoubleVector(int count, double high);

    void train(const std::vector<std::vector<double>>& trainInput, const std::vector<double>& trainLabel, double eta, int batchSize=32, int epochNumber=32);

    void test(const std::vector<std::vector<double>>& testInput, const std::vector<double>& testLabel);

    //void printNetwork();

    //std::vector<std::vector<double>> getRandomWeightMatrix(int inputs, int neurons, double low, double high);

    std::vector<double> sigmoid(const std::vector<double>& v);

    std::vector<double> sigmoidPrime(const std::vector<double>& v);

    std::vector<double> relu(const std::vector<double>& v);

    std::vector<double> reluPrime(const std::vector<double>& v);

    std::vector<double> getDoubleVector(int count, double value);

    double MSElossFunction(const std::vector<double>& output, const std::vector<double>& y);

    std::vector<double> MSElossDerivative(const std::vector<double>& output, const std::vector<double>& y);

    std::vector<double> SoftMax(const std::vector<double>& z);

    std::vector<double> SoftMaxDerivative(const std::vector<double>& z);

    double crossEntropyLoss(const std::vector<double>& output, const std::vector<double>& y);

    std::vector<double> crossEntropyLossDeriv(const std::vector<double>& output, const std::vector<double>& y);



    //std::vector<double> forwardPass(Layer layer, std::vector<double> input);

private:
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<std::vector<double>>> weights;

    //std::vector<NeuralLayer*> layers;
    int hiddenLayerCount;
    int neuronCount;
    int targetCount;
    int inputCount;
    int batchSize;
    int epochNumber;
    int currentBatchSize;

    //std::uniform_real_distribution<> dis{-1, 1};

    //std::default_random_engine engine;

    int totalLayerCount;
};

#endif // NEURALNETWORK_H
