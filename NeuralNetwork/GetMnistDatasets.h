#pragma once


#ifdef _WIN32
#define _WIN32_WINNT 0x0A00
#endif
#define ASIO_STANDALONE

#include <vector>
#include <asio.hpp>
#include <asio/ts/buffer.hpp>
#include <asio/ts/internet.hpp>

class GetMnistDatasets
{
	GetMnistDatasets();

	std::vector<std::vector<double>> getTrainingImages();

	std::vector<double> getTrainingLabels();

	std::vector<std::vector<double>> getTestImages();

	std::vector<double> getTestLabels();
};

