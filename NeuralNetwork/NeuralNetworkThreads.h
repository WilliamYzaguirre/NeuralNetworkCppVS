#pragma once

#include <vector>
#include "vectoroperations.h"
#include <math.h>
#include <random>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>

class NeuralNetworkThreads
{
public:
    NeuralNetworkThreads(int hiddenLayerCount, int neuronCount, int targetCount, int inputCount);

    ~NeuralNetworkThreads();

    std::vector<double> getRandomDoubleVector(int count, double high);

    void train(const std::vector<std::vector<double>>& trainInput, const std::vector<double>& trainLabel, double eta, int batchSize = 32, int epochNumber = 32);

    void test(const std::vector<std::vector<double>>& testInput, const std::vector<double>& testLabel);

    std::vector<double> relu(const std::vector<double>& v);

    std::vector<double> reluPrime(const std::vector<double>& v);

    std::vector<double> SoftMax(const std::vector<double>& z);

    double crossEntropyLoss(const std::vector<double>& output, const std::vector<double>& y);

private:
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> gradientb;
    std::vector<std::vector<std::vector<double>>> gradientw;

    int hiddenLayerCount;
    int neuronCount;
    int targetCount;
    int inputCount;
    int batchSize;
    int epochNumber;
    int currentBatchSize;

    int totalLayerCount;
};

