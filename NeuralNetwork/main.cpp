#include <iostream>
#include "neuralnetwork.h"
#include "NeuralNetworkIntrinsic.h"
#include <fstream>
#include <immintrin.h>
#include <random>
#include <chrono>
#include "NeuralNetworkThreads.h"
#include "vectoroperations.h"


int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
std::vector<std::vector<double>> readMnistImages(std::string path, int imageNumber, int dataOfImage)
{
    std::vector<std::vector<double>> ret;
    ret.resize(imageNumber, std::vector<double>(dataOfImage));
    std::ifstream file (path, std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        //std::cout << "ROWS: " << n_rows << std::endl;
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    ret[i][(n_rows*r)+c] = double(temp);
                }
            }
        }
        return ret;
    }
    else
    {
        std::cout << "Unable to open " << path << std::endl;
    }
}

std::vector<double> readMnistLabels(std::string path, int labelCount)
{
    std::vector<double> ret;
    ret.resize(labelCount);
    std::ifstream file(path, std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049)
        {
            throw std::runtime_error("Invalid MNIST label file!");
        }

        file.read((char *)&labelCount, sizeof(labelCount)), labelCount = reverseInt(labelCount);
        for (int i = 0; i < labelCount; ++i)
        {
            unsigned char temp = 0;
            file.read((char*)&temp, 1);
            ret[i] = (double)temp;
        }
        return ret;
    }
    else
    {
        std::cout << "Unable to open " << path << std::endl;
    }
}

void normalizeSpeedTests()
{
    std::srand((unsigned)time(NULL));
    std::vector<double> test(100000000, ((double)rand() / RAND_MAX) * (10));

    //non avx2
    auto start = std::chrono::high_resolution_clock::now();
    double total = 0;
    for (int i = 0; i < test.size(); ++i)
    {
        total += test[i] * test[i];
    }
    total = std::sqrt(total);
    for (int i = 0; i < test.size(); ++i)
    {
        test[i] = test[i] / total;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto standardDuration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Normal normal duration: " << standardDuration.count() << std::endl;

    //std::vector<double> test2(100000000, ((double)rand() / RAND_MAX) * (10));
    std::vector<double> test2{ 1.0,0,0,4.0 };
    // AVX2 normalization
    start = std::chrono::high_resolution_clock::now();
    total = 0;
    __m256d _temp;
    __m256d _total = _mm256_setzero_pd();
    for (int i = 0; i < test2.size(); i += 4)
    {
        _temp = _mm256_set_pd(test2[i], test2[i + 1], test2[i + 2], test2[i + 3]);
        _total = _mm256_fmadd_pd(_temp, _temp, _total);
    }
    total = _total.m256d_f64[0] + _total.m256d_f64[1] + _total.m256d_f64[2] + _total.m256d_f64[3];
    _total = _mm256_set1_pd(std::sqrt(total));
    for (int i = 0; i < test2.size(); i += 4)
    {
        _temp = _mm256_set_pd(test2[i], test2[i + 1], test2[i + 2], test2[i + 3]);
        _temp = _mm256_div_pd(_temp, _total);
        test2[i] = _temp.m256d_f64[3];
        test2[i + 1] = _temp.m256d_f64[2];
        test2[i + 2] = _temp.m256d_f64[1];
        test2[i + 3] = _temp.m256d_f64[0];
    }
    stop = std::chrono::high_resolution_clock::now();
    auto avx2Duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "AVX2 normal duration: " << avx2Duration.count() << std::endl;
    std::cout << "AVX2 is faster by " << standardDuration.count() - avx2Duration.count() << " milliseconds. LETS GOOOOO" << std::endl;
}



int main()
{
    std::vector<std::vector<double>> trainingImages = readMnistImages("C:\\Users\\willi\\Documents\\MNIST datasets\\train-images.idx3-ubyte", 60000, 784);
    std::vector<std::vector<double>> testImages = readMnistImages("C:\\Users\\willi\\Documents\\MNIST datasets\\t10k-images.idx3-ubyte", 10000, 784);
    std::vector<double> trainingLabels = readMnistLabels("C:\\Users\\willi\\Documents\\MNIST datasets\\train-labels.idx1-ubyte", 60000);
    std::vector<double> testLabels = readMnistLabels("C:\\Users\\willi\\Documents\\MNIST datasets\\t10k-labels.idx1-ubyte", 10000);


    std::vector<std::vector<double>> input{ {1, 2, 3, 4}, {4, 3, 2, 1} };
    std::vector<double> output{ 2, 3 };


    
    std::vector<std::vector<double>> shortTrain;
    std::vector<double> shortLabel;
    std::vector<int> indexes;
    for (int i = 0; i < 1000; ++i)
    {
        indexes.push_back(i);
        shortTrain.push_back(trainingImages[i]);
        shortLabel.push_back(trainingLabels[i]);
    }
    //std::random_shuffle(indexes.begin(), indexes.end());
    //for (auto value : indexes)
    //{
    //    shortTrain.push_back(trainingImages[value]);
    //    shortLabel.push_back(trainingLabels[value]);
    //}
    
    //int layerCount, int neuronCount, int targetCount, int inputCount
    NeuralNetworkIntrinsic nni{ 6, 128, 10, 784 };
    NeuralNetwork nn{ 6, 128, 10, 784 };
    NeuralNetworkThreads nnt{ 6, 128, 10, 784 };
    
    nnt.train(shortTrain, shortLabel, .1, 80, 50);

    // Below is a speed comparison with AVX2 and threads
    /*
    auto start1 = std::chrono::high_resolution_clock::now();
    nni.train(shortTrain, shortLabel, .1, 80, 50);
    auto stop1 = std::chrono::high_resolution_clock::now();
    auto avx2Duration = std::chrono::duration_cast<std::chrono::seconds>(stop1 - start1);
    nni.test(testImages, testLabels);

    std::cout << "\n######################\n" << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    nn.train(shortTrain, shortLabel, .1, 80, 50);
    auto stop2 = std::chrono::high_resolution_clock::now();
    auto normalDuration = std::chrono::duration_cast<std::chrono::seconds>(stop2 - start2);
    nn.test(testImages, testLabels);

    std::cout << "\n######################\n" << std::endl;

    auto start3 = std::chrono::high_resolution_clock::now();
    nnt.train(shortTrain, shortLabel, .1, 80, 50);
    auto stop3 = std::chrono::high_resolution_clock::now();
    auto threadDuration = std::chrono::duration_cast<std::chrono::seconds>(stop3 - start3);
    nnt.test(testImages, testLabels);

    std::cout << "Normal time: " << normalDuration.count() << " AVX2 time: " << avx2Duration.count() << " Thread time: " << threadDuration.count() << std::endl;

    */


    //nn.train(input, output, .1, 1, 1);


    //inputs, label, eta, batchsize, epoch


    return 0;
}

