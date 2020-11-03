#include <iostream>
#include "neuralnetwork.h"
#include "NeuralNetworkIntrinsic.h"
#include <fstream>

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

int main()
{

    std::vector<std::vector<double>> trainingImages = readMnistImages("D:\\MNIST datasets\\train-images.idx3-ubyte", 60000, 784);
    std::vector<std::vector<double>> testImages = readMnistImages("D:\\MNIST datasets\\t10k-images.idx3-ubyte", 10000, 784);
    std::vector<double> trainingLabels = readMnistLabels("D:\\MNIST datasets\\train-labels.idx1-ubyte", 60000);
    std::vector<double> testLabels = readMnistLabels("D:\\MNIST datasets\\t10k-labels.idx1-ubyte", 10000);


    std::vector<std::vector<double>> input{ {1, 2, 3, 4}, {4, 3, 2, 1} };
    std::vector<double> output{ 1, 2 };

    //int layerCount, int neuronCount, int targetCount, int inputCount
    //NeuralNetworkIntrinsic nni{4, 128, 10, 784};
    NeuralNetwork nn{ 1, 8, 1, 4 };
    NeuralNetworkIntrinsic nni{ 1, 8, 1, 4 };


    std::vector<std::vector<double>> shortTrain;
    std::vector<double> shortLabel;
    for (int i = 0; i < 500; ++i)
    {
        shortTrain.push_back(trainingImages[i*2]);
        shortLabel.push_back(trainingLabels[i*2]);
    }



    //nn.printNetwork();
    //nn.test(testImages, testLabels);
    //nn.train(shortTrain, shortLabel, .001, 60, 50);
    nn.train(input, output, .1, 1, 1);
    nni.train(input, output, .1, 1, 1);


    //inputs, label, eta, batchsize, epoch

    //nn.test(testImages, testLabels);


    return 0;
}

