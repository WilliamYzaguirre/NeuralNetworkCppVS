#ifndef VECTOROPERATIONS_H
#define VECTOROPERATIONS_H

#include <vector>
#include <iostream>
#include <math.h>

double vectorDotProduct(const std::vector<double>& v1, const std::vector<double>& v2) noexcept;

std::vector<double> vectorMatrixMult(const std::vector<std::vector<double>>& m, const std::vector<double>& v) noexcept;

std::vector<double> vectorAdd(const std::vector<double>& v1, const std::vector<double>& v2) noexcept;

std::vector<double> vectorSubtract(const std::vector<double>& v1, const std::vector<double>& v2) noexcept;

std::vector<double> hadamardVector(const std::vector<double>& v1, const std::vector<double>& v2) noexcept;

double vectorSum(const std::vector<double>& v1) noexcept;

std::vector<std::vector<double>> vectorTransposeMult(const std::vector<double>& v1, const std::vector<double>& v2) noexcept;

std::vector<std::vector<double>> matrixTranspose(const std::vector<std::vector<double>>& m) noexcept;

void normalizeVector(std::vector<double>& v);

std::vector<double> averageVectors(const std::vector<std::vector<double>>& vectors) noexcept;

#endif // VECTOROPERATIONS_H
