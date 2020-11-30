#pragma once

#include <vector>
#include "vectoroperations.h"
#include <math.h>
#include <random>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <thread>
#include <mutex>

class NeuralNetworkThreads
{
public:
    NeuralNetworkThreads(int hiddenLayerCount, int neuronCount, int targetCount, int inputCount);

    ~NeuralNetworkThreads();

    void BackPropogate(std::vector<double> activation, double label);

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

    std::vector<std::vector<std::vector<double>>> totalWeightGradient;
    std::vector<std::vector<double>> totalBiasGradient;

    std::vector<double> costs;


    int hiddenLayerCount;
    int neuronCount;
    int targetCount;
    int inputCount;
    int batchSize;
    int epochNumber;
    int currentBatchSize;

    int totalLayerCount;

    std::mutex mux;
};

