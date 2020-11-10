#include "NeuralNetworkIntrinsic.h"


/*
TODO: pushing back to z and a is awful. Needs to be indexable for assigning. Either shape zs and activations when you make
bias and weights. Or do it at the begining of the train function. Or just make everything arrays. Might be slightly faster.
Why didn't I do this to start with. So much pain. Also write unit tests dumb dumb
*/

/*

FOR THE LOVE OF GOD ONLY USE THIS IF THE NEURON COUNT IN ALL OF THE LAYERS (except the output in this case) ARE DIVISIBLE BY 4 AND YOU HAVE
A PROCESSOR THAT SUPPORTS AVX 4

*/

NeuralNetworkIntrinsic::NeuralNetworkIntrinsic(int hiddenLayerCount, int neuronCount, int targetCount, int inputCount)
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

        }

        // Otherwise, we make a hidden layer

        else
        {
            // Hidden layers have neuronCount biases

            //biases.push_back(getRandomDoubleVector(neuronCount, 1));

            biases.push_back(std::vector<double>(neuronCount, .1));

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
                //layer.push_back(getRandomDoubleVector(inputCount, 1));
                layer.push_back(std::vector<double>(inputCount, 1));

            }

            weights.push_back(layer);
        }

        //  Then target layer

        else if (i == totalLayerCount - 1)
        {
            // First, make the layer vector

            std::vector<std::vector<double>> layer;

            // We need to give the target layer targetCount vectors of neuronCount size

            for (int j = 0; j < targetCount; ++j)
            {
                //layer.push_back(getRandomDoubleVector(neuronCount, 1));
                layer.push_back(std::vector<double>(neuronCount, 2));

            }

            weights.push_back(layer);
        }
        // Otherwise, we give the layer neuronCount vectors of neuronCount size

        else
        {
            // Make the layer

            std::vector<std::vector<double>> layer;

            for (int j = 0; j < neuronCount; ++j)
            {
                //layer.push_back(getRandomDoubleVector(neuronCount, 1));
                layer.push_back(std::vector<double>(neuronCount, 3));

            }

            weights.push_back(layer);
        }
    }

}

NeuralNetworkIntrinsic::~NeuralNetworkIntrinsic()
{

}

std::vector<double> NeuralNetworkIntrinsic::getRandomDoubleVector(int count, double high)
{
    std::vector<double> ret;
    for (int i = 0; i < count; ++i)
    {
        double random = (((double)rand() / RAND_MAX) * (high * 2) - high);
        ret.push_back(random);
    }
    return ret;
}


void NeuralNetworkIntrinsic::train(const std::vector<std::vector<double>>& trainInput, const std::vector<double>& trainLabel, double eta, int batchSize, int epochNumber)
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


                std::vector<std::vector<double>> zs; // 2D matrix to store all z's. z = weight . activation + b
                std::vector<std::vector<double>> activations; // 2D matrix to store all activations. activation = acticationFunction(z)

                // Store the gradient for a single pass, then gets added to total gradients in batch
                std::vector<std::vector<std::vector<double>>> gradientw;
                std::vector<std::vector<double>> gradientb;

                // std::vector<std::vector<double>> deltas;             // debug

                activations.push_back(activation);

                // z = a * w + a * w + ... + a * w + b
                
                // 64-bit double registers
                __m256d _a, _w, _z, _b, _mask1, _total;

                __m256d _zero = _mm256_setzero_pd();

                // Forward pass
                // Start with hidden layers
                // Using intrinsic to calculate 4 outputs z, a at once

                for (int layer = 0; layer < hiddenLayerCount; ++layer)
                {
                    std::vector<double> z(activation.size(), 0);

                    // Need to reset total each layer to normalize all activations
                    __m256d _total = _mm256_setzero_pd();

                    for (int neur = 0; neur < weights[layer].size(); neur += 4)
                    {
                        _z = _mm256_set1_pd(0.0);          // z = |0|0|0|0|

                        for (int values = 0; values < weights[layer][neur].size(); ++values)
                        {
                            // _w = weights[layer][neuron0,1,2,3][value in neuron]
                            _w = _mm256_set_pd(weights[layer][neur][values], weights[layer][neur+1][values], weights[layer][neur+2][values], weights[layer][neur+3][values]);

                            // _a = |activation0|activation0|activation0|activation0|
                            _a = _mm256_set1_pd(activation[values]);

                            _z = _mm256_fmadd_pd(_a, _w, _z);  // z = a * w + z
                        }

                        // _b = |bias0|bias1|bias2|bias3|
                        _b = _mm256_set_pd(biases[layer][neur], biases[layer][neur + 1], biases[layer][neur + 2], biases[layer][neur + 3]);


                        _z = _mm256_add_pd(_z, _b);      // z = a * w + b

// I don't know why, but I can't get these lines to work. Am I secretly using linux?
//#if defined(_WIN64)
                        z[neur] = _z.m256d_f64[3];
                        z[neur + 1] = _z.m256d_f64[2];
                        z[neur + 2] = _z.m256d_f64[1];
                        z[neur + 3] = _z.m256d_f64[0];                        
//#endif

                        // Now we take the 4 z's and apply the activation funcction (relu) to get 4 a's
                        // The z's in the same layer have no dependency on each other to get it's activation
                        
                        // You have done a terrible job commenting this idiot
                        // Beginning the relu (Good start)
                        // a = max(0, z)
                        // if (z > 0) then a = z
                        // else a = 0

                        _mask1 = _mm256_cmp_pd(_z, _zero, _CMP_GT_OQ);

                        // if true, then a[true] = |11...11|
                        // if false, then a[false] = |00...00|
                        // logically and _mask1 with _z to get activation
                        // When you and with all 1's, that block will be the z
                        // When you and with all 0's, it becomes zero

                        _a = _mm256_and_pd(_z, _mask1);

                        // oh god, now we have to worry about normalizing...
                        // we can either wait till the end and add 4 at a time, then sum the resulting 4
                        // or we just keep a running total at this point
                        



                    }

                    //z = vectorAdd(vectorMatrixMult(weights[layer], activation), biases[layer]);
                 

                    //std::vector<double> newActivation = sigmoid(z);
                    activation = relu(z);
                    normalizeVector(activation);
                    activations.push_back(activation);
                }

                //zs.push_back(z);

                for (auto zi : zs)
                {
                    for (auto zj : zi)
                    {
                        std::cout << zj << ", ";
                    }
                    std::cout << std::endl;
                }

                // Output layer needs different activation function for binary classification (mnist)
            //    z = vectorAdd(vectorMatrixMult(weights[totalLayerCount - 1], activation), biases[totalLayerCount - 1]);
            //    zs.push_back(z);
                // Output layer activation function goes here
            //    activation = SoftMax(z);
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
                gradientb.push_back(delta);

                // deltas.push_back(delta);

                gradientw.push_back(vectorTransposeMult(delta, activations[activations.size() - 2]));

                for (int i = 2; i < totalLayerCount + 1; ++i)
                {
            //        z = zs[zs.size() - i];
                    //std::vector<double> sp = sigmoidPrime(z);
            //        std::vector<double> sp = reluPrime(z);

            //        delta = hadamardVector(vectorMatrixMult(matrixTranspose(weights[weights.size() - i + 1]), delta), sp);
            
                    gradientb.push_back(delta);
                    gradientw.push_back(vectorTransposeMult(delta, activations[activations.size() - i - 1]));
                    // deltas.push_back(delta);
                }

                // Reverse them since I stored them in reverse order
                std::reverse(gradientb.begin(), gradientb.end());
                std::reverse(gradientw.begin(), gradientw.end());


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
                // std::vector<double> test = getDoubleVector(totalBiasGradient[i].size(), (1.0/currentBatchSize));
                // std::vector<double> hadTest = hadamardVector(totalBiasGradient[i], test);
                //normalizeVector(hadTest);
                totalBiasGradient[i] = hadamardVector(totalBiasGradient[i], getDoubleVector(totalBiasGradient[i].size(), (1.0 / currentBatchSize)));
                normalizeVector(totalBiasGradient[i]);
            }

            for (int i = 0; i < int(totalWeightGradient.size()); ++i)
            {
                for (int j = 0; j < int(totalWeightGradient[i].size()); ++j)
                {
                    totalWeightGradient[i][j] = hadamardVector(totalWeightGradient[i][j], getDoubleVector(totalWeightGradient[i][j].size(), (1.0 / currentBatchSize)));
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

void NeuralNetworkIntrinsic::test(const std::vector<std::vector<double>>& testInput, const std::vector<double>& testLabel)
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

std::vector<double> NeuralNetworkIntrinsic::sigmoid(const std::vector<double>& v)
{
    std::vector<double> ret;
    for (auto item : v)
    {
        ret.push_back(1.0 / (1 + std::exp(-item)));
    }
    return ret;
}

std::vector<double> NeuralNetworkIntrinsic::sigmoidPrime(const std::vector<double>& v)
{
    std::vector<double> ret = hadamardVector(sigmoid(v), vectorSubtract(getDoubleVector(v.size(), 1.0), sigmoid(v))); //sig(v) * (1 - sig(v)
    return ret;
}

std::vector<double> NeuralNetworkIntrinsic::relu(const std::vector<double>& v)
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

std::vector<double> NeuralNetworkIntrinsic::reluPrime(const std::vector<double>& v)
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

std::vector<double> NeuralNetworkIntrinsic::getDoubleVector(int count, double value)
{
    std::vector<double> ret;
    for (int i = 0; i < count; ++i)
    {
        ret.push_back(value);
    }
    return ret;
}

double NeuralNetworkIntrinsic::MSElossFunction(const std::vector<double>& output, const std::vector<double>& y)
{
    std::vector<double> diff = vectorSubtract(output, y);
    diff = hadamardVector(diff, getDoubleVector(diff.size(), 2.0 / currentBatchSize));
    return (vectorSum(hadamardVector(diff, diff)));
}

std::vector<double> NeuralNetworkIntrinsic::MSElossDerivative(const std::vector<double>& output, const std::vector<double>& y)
{
    std::vector<double> diff = vectorSubtract(output, y);
    return (hadamardVector(diff, getDoubleVector(diff.size(), 2.0 / currentBatchSize)));
}

std::vector<double> NeuralNetworkIntrinsic::SoftMax(const std::vector<double>& z)
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


double NeuralNetworkIntrinsic::crossEntropyLoss(const std::vector<double>& output, const std::vector<double>& y)
{
    std::vector<double> temp;
    for (auto val : output)
    {
        temp.push_back(log(val));
    }
    double ret = vectorDotProduct(temp, y);
    return (-1.0 / currentBatchSize) * ret;
}