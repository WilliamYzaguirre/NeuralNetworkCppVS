#include "NeuralNetworkIntrinsic.h"


/*
TODO: pushing back to z and a is awful. Needs to be indexable for assigning. Either shape zs and activations when you make
bias and weights. Or do it at the begining of the train function. Or just make everything arrays. Might be slightly faster.
Why didn't I do this to start with. So much pain. Also write unit tests dumb dumb
UPDATE: Units tests are being janky, need to figure it out. Made everything indexable. Arrays are not faster. Stop doubting 
people who design this stuff
TODO: Maybe look into making forward pass a funciton that can take in different types of activations, then making each layer
an object that hold it's own info, like neuron count, activation function type, etc... I believe that's what pytorch does.
Seems possibly more readable. Ironically, this is literally the design you built like 3/4 of the way then switched to this
less modular design. We'll see how that plays out in the long run
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

                // I'm gonna cheat a little here with normalization. I know every image will have a min of 0, and a
                // max of 255. So I'm gonna use those for min-max scaling, without having to iterate the whole vector
                
                __m256d _temp;
                __m256d _divisor = _mm256_set1_pd(255.0);
                for (int i = 0; i < activation.size(); i += 4)
                {
                    _temp = _mm256_set_pd(activation[i], activation[i + 1], activation[i + 2], activation[i + 3]);
                    _temp = _mm256_div_pd(_temp, _divisor);
                    activation[i] = _temp.m256d_f64[0];
                    activation[i + 1] = _temp.m256d_f64[1];
                    activation[i + 2] = _temp.m256d_f64[2];
                    activation[i + 3] = _temp.m256d_f64[3];
                }
                

                std::vector<std::vector<double>> zs; // 2D matrix to store all z's. z = weight . activation + b
                std::vector<std::vector<double>> activations; // 2D matrix to store all activations. activation = acticationFunction(z)

                // Store the gradient for a single pass, then gets added to total gradients in batch
                //std::vector<std::vector<std::vector<double>>> gradientw;
                //std::vector<std::vector<double>> gradientb;

                // std::vector<std::vector<double>> deltas;             // debug

                activations.push_back(activation);

                // z = a * w + a * w + ... + a * w + b
                
                // 64-bit double registers
                __m256d _a, _w, _z, _b, _mask1, _total;

                __m256d _zero = _mm256_setzero_pd();

                std::vector<double> z;
                std::vector<double> newActivation;


                // Forward pass
                // Start with hidden layers
                // Using intrinsic to calculate 4 outputs z, a at once

                for (int layer = 0; layer < hiddenLayerCount; ++layer)
                {
                    // at the start of each new layer, we need a z and newActivation thats as large as the number of neurons in that layers
                    // note I'm using resize and not assign, this leaves the values the same, but changes size. I think this is
                    // fine because I'm assigning values later, not pushing. Also I assume resize has to be slightly faster
                    z.resize(biases[layer].size(), 0);
                    newActivation.resize(biases[layer].size(), 0);

                    // Need to reset total each layer to normalize all activations
                    //__m256d _total = _mm256_setzero_pd();

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
                        z[neur] = _z.m256d_f64[0];
                        z[neur + 1] = _z.m256d_f64[1];
                        z[neur + 2] = _z.m256d_f64[2];
                        z[neur + 3] = _z.m256d_f64[3];                        
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
                        // so with normalizing, I'm going to wait until the end and only normalize the activations
                        // since z's will only be used later in reluPrime, which doesn't really matter if it's large or not
                        //
                        // we can either wait till the end and add 4 at a time, then sum the resulting 4
                        // or we just keep a running total at this point
                        
                        newActivation[neur] = _a.m256d_f64[0];
                        newActivation[neur + 1] = _a.m256d_f64[1];
                        newActivation[neur + 2] = _a.m256d_f64[2];
                        newActivation[neur + 3] = _a.m256d_f64[3];


                    }
                    // normalize with AVX2 normalization funciton, I can maybe speed this up later by writing this function in-line
                    // and keeping a running total of the activations as I make them for the normalization function.
                    // also look into standardization and using min-max normalization instead
                    zs.push_back(z);
                    normalizeVectorAVX2(newActivation);

                    // use move semantics
                    activation = std::move(newActivation);

                    activations.push_back(activation);
                }

                // Output layer needs different activation function for binary classification (mnist)
                // Umm I don't know if using AVX2 on a single layer like this is actually worth it... but let's do it for fun. MNIST has 10 outputs idiot
                // Don't use AVX2
                z = vectorAdd(vectorMatrixMult(weights[totalLayerCount - 1], activation), biases[totalLayerCount - 1]);
                zs.push_back(z);
                // Output layer activation function goes here
                // Honestly, for mnist, I know there's only 10 values here, I'm not gonna use AVX2 for this
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
                // Once again, not worth it to use AVX2

                std::vector<double> delta = vectorSubtract(activations[activations.size() - 1], y);
                //gradientb.push_back(delta);
                gradientb[gradientb.size() - 1] = delta;

                // deltas.push_back(delta);
                // to get weight gradient, simply multiply the last activation layer by delta
                // Once again, I don't think that's worth it for AVX2

                //gradientw.push_back(vectorTransposeMult(delta, activations[activations.size() - 2]));
                gradientw[gradientw.size() - 1] = vectorTransposeMult(delta, activations[activations.size() - 2]);

                for (int i = 2; i < totalLayerCount + 1; ++i)
                {
                    // Now we're working with larger sets of data again, lets get back to the AVX2
                    // start by iterating through the z's, once again, 4 at a time
                    // Backprop is a little harder, so let's think about this
                    // Start by applying reluPrime to z (0 if <= 0, else 1)
                    // _z = | z[0] | z[1] | z[2] | z[3] |
                    // _z GT _0 = _mask1
                    // _mask1 = | 0000 | 1111 | 0000 | 1111 |
                    // _mask2 = | 0001 | 0001 | 0001 | 0001 |
                    // _sp = _mask1 and _mask2
                    // Then get delta
                    // delta[l] = w[l+1]T*delta[l+1] o activationDerivative(z(L))
                    // 
                    // ... We might come back to this after threading

                    z = zs[zs.size() - i];
                    //std::vector<double> sp = sigmoidPrime(z);
                    std::vector<double> sp = reluPrime(z);

                    delta = hadamardVector(vectorMatrixMult(matrixTranspose(weights[weights.size() - i + 1]), delta), sp);
            
                    //gradientb.push_back(delta);
                    gradientb[gradientb.size() - i] = delta;
                    //gradientw.push_back(vectorTransposeMult(delta, activations[activations.size() - i - 1]));
                    gradientw[gradientw.size() - i] = vectorTransposeMult(delta, activations[activations.size() - i - 1]);
                    // deltas.push_back(delta);
                }

                // Reverse them since I stored them in reverse order
                //std::reverse(gradientb.begin(), gradientb.end());
                //std::reverse(gradientw.begin(), gradientw.end());


                // Update the total gradients per batch
                // I don't think doing this with avx2 would be faster
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

            // I think I can combine the normalizing and updating into one for loop and use AVX2 for gainz

            // Normalize the total gradient vectors in a wjole batch
            for (int i = 0; i < int(totalBiasGradient.size()); ++i)
            {
                // to start we take the average of the whole batch
                // _b = | b[i][0] | b[i][1] | b[i][2] | b[i][3] |
                // _avg = | 1 / batchsize | 1 / batchsize | 1 / batchsize | 1 / batchsize |
                // _b = 
                totalBiasGradient[i] = hadamardVector(totalBiasGradient[i], getDoubleVector(totalBiasGradient[i].size(), (1.0 / currentBatchSize)));
                if (i != totalBiasGradient.size() - 1)
                {
                    normalizeVectorAVX2(totalBiasGradient[i]);
                }
                else
                {
                    normalizeVector(totalBiasGradient[i]);
                }
            }

            for (int i = 0; i < int(totalWeightGradient.size()); ++i)
            {
                for (int j = 0; j < int(totalWeightGradient[i].size()); ++j)
                {
                    totalWeightGradient[i][j] = hadamardVector(totalWeightGradient[i][j], getDoubleVector(totalWeightGradient[i][j].size(), (1.0 / currentBatchSize)));
                    normalizeVectorAVX2(totalWeightGradient[i][j]);
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
                        weights[i][j][h] -= eta * totalWeightGradient[i][j][h];
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

void NeuralNetworkIntrinsic::normalizeVectorAVX2(std::vector<double>& v)
{
    double total = 0;
    __m256d _temp;
    __m256d _total = _mm256_setzero_pd();
    for (int i = 0; i < v.size(); i += 4)
    {
        _temp = _mm256_set_pd(v[i], v[i + 1], v[i + 2], v[i + 3]);
        _total = _mm256_fmadd_pd(_temp, _temp, _total);
    }
    total = _total.m256d_f64[0] + _total.m256d_f64[1] + _total.m256d_f64[2] + _total.m256d_f64[3];
    total = std::sqrt(total);
    _total = _mm256_set1_pd(total);
    for (int i = 0; i < v.size(); i += 4)
    {
        _temp = _mm256_set_pd(v[i], v[i + 1], v[i + 2], v[i + 3]);
        _temp = _mm256_div_pd(_temp, _total);
        v[i] = _temp.m256d_f64[0];
        v[i + 1] = _temp.m256d_f64[1];
        v[i + 2] = _temp.m256d_f64[2];
        v[i + 3] = _temp.m256d_f64[3];
    }
}
