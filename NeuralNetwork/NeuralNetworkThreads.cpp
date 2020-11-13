#include "NeuralNetworkThreads.h"

NeuralNetworkThreads::NeuralNetworkThreads(int hiddenLayerCount, int neuronCount, int targetCount, int inputCount)
	:hiddenLayerCount{ hiddenLayerCount }, neuronCount{ neuronCount }, targetCount{ targetCount }, inputCount{ inputCount }
{
    totalLayerCount = hiddenLayerCount + 1;

    // Initialize bias. This is a vector of layerCount + 1 vectors
    // We use layerCount + 1 because we need to make the target layer as well
    // I don't need to make an input layer because it has no weights or bias. It's just data

    for (int i = 0; i < totalLayerCount; ++i)
    {
        // Start with target layer since it's special. We want to catch it before the others.

        if (i == totalLayerCount - 1)
        {
            // Target layer gets targetCount biases

            //biases.push_back(getRandomDoubleVector(targetCount, 1));

            biases.push_back(std::vector<double>(targetCount, .1));  //target count .1
            gradientb.push_back(std::vector<double>(targetCount, 0));

        }

        // Otherwise, we make a hidden layer

        else
        {
            // Hidden layers have neuronCount biases

            //biases.push_back(getRandomDoubleVector(neuronCount, 1));

            biases.push_back(std::vector<double>(neuronCount, .1));
            gradientb.push_back(std::vector<double>(neuronCount, 0));
        }
    }

    // Initialize the weights. This is a vector of layer + 1 vectors of neurons
    std::srand((unsigned)time(NULL));

    for (int i = 0; i < totalLayerCount; ++i)
    {
        // Start with first layer since it has unique weight counts (inputCount)

        if (i == 0)
        {
            // First, make the layer vector

            std::vector<std::vector<double>> layer;

            // We need to give the target layer targetCount vectors of neuronCount size

            for (int j = 0; j < neuronCount; ++j)
            {
                layer.push_back(getRandomDoubleVector(inputCount, 1));
                //layer.push_back(std::vector<double>(inputCount, 1));

            }

            weights.push_back(layer);
            gradientw.push_back(layer);
        }

        //  Then target layer

        else if (i == totalLayerCount - 1)
        {
            // First, make the layer vector

            std::vector<std::vector<double>> layer;

            // We need to give the target layer targetCount vectors of neuronCount size

            for (int j = 0; j < targetCount; ++j)
            {
                layer.push_back(getRandomDoubleVector(neuronCount, 1));
                //layer.push_back(std::vector<double>(neuronCount, 2));

            }

            weights.push_back(layer);
            gradientw.push_back(layer);
        }
        // Otherwise, we give the layer neuronCount vectors of neuronCount size

        else
        {
            // Make the layer

            std::vector<std::vector<double>> layer;

            for (int j = 0; j < neuronCount; ++j)
            {
                layer.push_back(getRandomDoubleVector(neuronCount, 1));
                //layer.push_back(std::vector<double>(neuronCount, 3));

            }

            weights.push_back(layer);
            gradientw.push_back(layer);
        }
    }
}

NeuralNetworkThreads::~NeuralNetworkThreads()
{

}

std::vector<double> NeuralNetworkThreads::getRandomDoubleVector(int count, double high)
{
    std::vector<double> ret;
    for (int i = 0; i < count; ++i)
    {
        double random = (((double)rand() / RAND_MAX) * (high * 2) - high);
        ret.push_back(random);
    }
    return ret;
}

void NeuralNetworkThreads::train(const std::vector<std::vector<double>>& trainInput, const std::vector<double>& trainLabel, double eta, int batchSize, int epochNumber)
{
    double totalPoints = trainInput.size();

    for (int epoch = 0; epoch < epochNumber; ++epoch)
    {
        // batchcount is total number of batches to run

        int batchCount = ceil(totalPoints / batchSize);

        std::vector<double> costs;

        for (int iteration = 0; iteration < batchCount; ++iteration)
        {
            // Current batch size is variable because last set might not have enough for whole batch

            currentBatchSize = batchSize;
            if (totalPoints - batchSize * iteration < batchSize)
            {
                currentBatchSize = totalPoints - batchSize * iteration;
            }

            std::vector<std::vector<std::vector<double>>> totalWeightGradient;
            for (auto layer : weights)
            {
                std::vector<std::vector<double>> row;
                for (auto neuron : layer)
                {
                    //row.push_back(getDoubleVector(neuron.size(), 0));
                    row.push_back(std::vector<double>(neuron.size(), 0));
                }
                totalWeightGradient.push_back(row);
            }

            std::vector<std::vector<double>> totalBiasGradient;
            for (auto layer : biases)
            {
                //totalBiasGradient.push_back(getDoubleVector(layer.size(), 0));
                totalBiasGradient.push_back(std::vector<double>(layer.size(), 0));
            }

            for (int batch = 0; batch < currentBatchSize; ++batch)
            {

                // Keeps track of where we are in the total dataset
                int currentInput = (iteration * batchSize) + batch;

                // First input to go into the forward pass. Named activation for ease in forward pass
                std::vector<double> activation = trainInput[currentInput];
                minMaxNormalizeVector(activation, .1, .9);


                std::vector<std::vector<double>> zs; // 2D matrix to store all z's. z = weight . activation + b
                std::vector<std::vector<double>> activations; // 2D matrix to store all activations. activation = acticationFunction(z)

                // std::vector<std::vector<double>> deltas;             // debug

                activations.push_back(activation);

                // Forward pass
                // Start with hidden layers
                for (int i = 0; i < hiddenLayerCount; ++i)
                {
                    std::vector<double> z = vectorAdd(vectorMatrixMult(weights[i], activation), biases[i]);
                    //normalizeVector(z);
                    //minMaxNormalizeVector(z);
                    zs.push_back(z);
                    //std::vector<double> newActivation = sigmoid(z);
                    activation = relu(z);
                    minMaxNormalizeVector(activation, .1, .9);
                    activations.push_back(activation);
                }



                // Output layer needs different activation function for binary classification (mnist)
                std::vector<double> z = vectorAdd(vectorMatrixMult(weights[totalLayerCount - 1], activation), biases[totalLayerCount - 1]);
                // There is a significant decrease in learning when you normalize last z. DON'T
                zs.push_back(z);
                // Output layer activation function goes here
                activation = SoftMax(z);
                //std::vector<double> newActivation = relu(z);
                activations.push_back(activation);

                // Back pass
                // Get correct vector

                std::vector<double> y;
                for (int i = 0; i < targetCount; ++i)
                {
                    if (i == trainLabel[currentInput])
                    {
                        y.push_back(1);
                    }
                    else
                    {
                        y.push_back(0);
                    }
                }

                // delta = costDerivative(a(L) - y)*activationDerivative(z(L))
                // dC/db = costDerivative(a(L) - y)*activationDerivative(z(L))
                // dC/dw = a(L-1)*costDerivative(a(L) - y)*activationDerivative(z(L))

                // Using cross entropy loss and softmax for output
                double cost = crossEntropyLoss(activations[activations.size() - 1], y);
                costs.push_back(cost);

                // Use this for MSE and sigmoid output
                // std::vector<double> delta = hadamardVector(MSElossDerivative(activations[activations.size() - 1], y), sigmoidPrime(zs[zs.size() - 1]));

                // Use this for MSE and relu output
                // std::vector<double> delta = hadamardVector(costDerivative(activations[activations.size() - 1], y), reluPrime(zs[zs.size() - 1]));

                // Use this for softmax and cross entropy

                std::vector<double> delta = vectorSubtract(activations[activations.size() - 1], y);
                gradientb[gradientb.size() - 1] = delta;

                // deltas.push_back(delta);

                gradientw[gradientw.size() - 1] = vectorTransposeMult(delta, activations[activations.size() - 2]);

                for (int i = 2; i < totalLayerCount + 1; ++i)
                {
                    std::vector<double> z = zs[zs.size() - i];
                    //std::vector<double> sp = sigmoidPrime(z);
                    std::vector<double> sp = reluPrime(z);

                    delta = hadamardVector(vectorMatrixMult(matrixTranspose(weights[weights.size() - i + 1]), delta), sp);

                    gradientb[gradientb.size() - i] = delta;
                    gradientw[gradientw.size() - i] = vectorTransposeMult(delta, activations[activations.size() - i - 1]);
                    // deltas.push_back(delta);
                }


                // Update the total gradients per batch
                for (int i = 0; i < int(totalBiasGradient.size()); ++i)
                {
                    for (int j = 0; j < int(totalBiasGradient[i].size()); ++j)
                    {
                        totalBiasGradient[i][j] += gradientb[i][j];
                    }
                }

                for (int i = 0; i < int(totalWeightGradient.size()); ++i)
                {
                    for (int j = 0; j < int(totalWeightGradient[i].size()); ++j)
                    {
                        for (int h = 0; h < int(totalWeightGradient[i][j].size()); ++h)
                        {
                            totalWeightGradient[i][j][h] += gradientw[i][j][h];
                        }
                    }
                }


            }

            // Normalize the total gradient vectors in a wjole batch
            for (int i = 0; i < int(totalBiasGradient.size()); ++i)
            {
                totalBiasGradient[i] = hadamardVector(totalBiasGradient[i], std::vector<double>(totalBiasGradient[i].size(), (1.0 / currentBatchSize)));
                normalizeVector(totalBiasGradient[i]);
            }

            for (int i = 0; i < int(totalWeightGradient.size()); ++i)
            {
                for (int j = 0; j < int(totalWeightGradient[i].size()); ++j)
                {
                    totalWeightGradient[i][j] = hadamardVector(totalWeightGradient[i][j], std::vector<double>(totalWeightGradient[i][j].size(), (1.0 / currentBatchSize)));
                    normalizeVector(totalWeightGradient[i][j]);
                }
            }

            // Update the weights and biases: new = old - stepSize*biasGradient
            for (int i = 0; i < int(biases.size()); ++i)
            {
                for (int j = 0; j < int(biases[i].size()); ++j)
                {
                    biases.at(i).at(j) -= eta * totalBiasGradient[i][j];
                }
            }

            for (int i = 0; i < int(weights.size()); ++i)
            {
                for (int j = 0; j < int(weights[i].size()); ++j)
                {
                    for (int h = 0; h < int(weights[i][j].size()); ++h)
                    {
                        weights[i][j][h] = weights[i][j][h] - eta * totalWeightGradient[i][j][h];
                    }
                }
            }

            double realCost = 0;
            for (auto val : costs)
            {
                realCost += val;
            }
            std::cout << "Cost after batch: " << realCost << std::endl;
            costs.clear();
        }
        std::cout << "Epoch " << epoch << " Complete." << std::endl;
    }
}

void NeuralNetworkThreads::test(const std::vector<std::vector<double>>& testInput, const std::vector<double>& testLabel)
{
    int correct = 0;
    int incorrect = 0;
    std::vector<double> z;
    for (int input = 0; input < int(testInput.size()); ++input)
    {
        std::vector<double> activation = testInput[input];
        for (int i = 0; i < hiddenLayerCount; ++i)
        {
            z = vectorAdd(vectorMatrixMult(weights[i], activation), biases[i]);
            //activation = sigmoid(z);
            activation = relu(z);
            normalizeVector(activation);
        }

        z = vectorAdd(vectorMatrixMult(weights[totalLayerCount - 1], activation), biases[totalLayerCount - 1]);
        //activation = sigmoid(z);
        //activation = relu(z);
        activation = SoftMax(z);

        int index = 0;
        double max = activation[0];
        for (int i = 1; i < int(activation.size()); ++i)
        {
            if (activation[i] > max)
            {
                index = i;
                max = activation[i];
            }
        }
        if (index == testLabel[input])
        {
            correct++;
        }
        else
        {
            incorrect++;
        }
    }
    std::cout << "Correct: " << correct << ". Incorrect: " << incorrect << std::endl;
}

std::vector<double> NeuralNetworkThreads::relu(const std::vector<double>& v)
{
    std::vector<double> ret;
    for (auto item : v)
    {
        if (item > 0)
        {
            ret.push_back(item);
        }
        else
        {
            ret.push_back(0);
        }
    }
    return ret;
}

std::vector<double> NeuralNetworkThreads::reluPrime(const std::vector<double>& v)
{
    std::vector<double> ret;
    for (auto item : v)
    {
        if (item <= 0)
        {
            ret.push_back(0);
        }
        else
        {
            ret.push_back(1);
        }
    }
    return ret;
}

std::vector<double> NeuralNetworkThreads::SoftMax(const std::vector<double>& z)
{
    double total = 0;
    double max = 0;
    for (auto val : z)
    {
        if (val > max)
        {
            max = val;
        }
    }
    std::vector<double> ret;
    for (auto val : z)
    {
        total += std::exp(val - max);
    }
    for (auto val : z)
    {
        ret.push_back(std::exp(val - max) / total);
    }
    return ret;
}

double NeuralNetworkThreads::crossEntropyLoss(const std::vector<double>& output, const std::vector<double>& y)
{
    std::vector<double> temp;
    for (auto val : output)
    {
        temp.push_back(log(val));
    }
    double ret = vectorDotProduct(temp, y);
    return (-1.0 / currentBatchSize) * ret;
}
