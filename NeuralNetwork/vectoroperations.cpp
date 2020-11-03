#include "vectoroperations.h"

double vectorDotProduct(const std::vector<double>& v1, const std::vector<double>& v2) noexcept
{
    if (v2.size() != v1.size())
    {
        std::cout << "Dot error. Vector sizes do not match. Size 1: " << v1.size() << " Size 2: " << v2.size() << std::endl;
        return 0;
    }
    else
    {
        double total = 0;
        for (int i = 0; i < int(v2.size()); ++i)
        {
            total += v2[i] * v1[i];
        }
        return total;
    }
}

std::vector<double> vectorMatrixMult(const std::vector<std::vector<double>>& m, const std::vector<double>& v) noexcept
{
    std::vector<double> ret;
    for (auto mvector : m)
    {
        if (v.size() != mvector.size())
        {
            std::cout << "Matrix and Vector sizes do not match. Vector 1: " << v.size() << " Row 2: " << mvector.size() << std::endl;
            return ret;
        }
        else
        {
            ret.push_back(vectorDotProduct(mvector, v));
        }
    }
    return ret;
}


std::vector<double> vectorAdd(const std::vector<double>& v1, const std::vector<double>& v2) noexcept
{
    std::vector<double> ret;
    if (v2.size() != v1.size())
    {
        std::cout << "Add error. Input and Weight sizes do not match. Size 1: " << v1.size() << " Size 2: " << v2.size() << std::endl;
        return ret;
    }
    else
    {
        for (int i = 0; i < (int)v2.size(); ++i)
        {
            ret.push_back(v1[i] + v2[i]);
        }
        return ret;
    }
}

std::vector<double> vectorSubtract(const std::vector<double>& v1, const std::vector<double>& v2) noexcept
{
    std::vector<double> ret;
    if (v2.size() != v1.size())
    {
        std::cout << "Sub error. Input and Weight sizes do not match. Size 1: " << v1.size() << " Size 2: " << v2.size() << std::endl;
        return ret;
    }
    else
    {
        for (int i = 0; i < (int)v2.size(); ++i)
        {
            ret.push_back(v1[i] - v2[i]);
        }
        return ret;
    }
}

std::vector<double> hadamardVector(const std::vector<double>& v1, const std::vector<double>& v2) noexcept
{
    std::vector<double> ret;
    if (v2.size() != v1.size())
    {
        std::cout << "Hadamard error. Input and Weight sizes do not match. Size 1: " << v1.size() << " Size 2: " << v2.size() << std::endl;
        return ret;
    }
    else
    {
        for (int i = 0; i < (int)v2.size(); ++i)
        {
            ret.push_back(v1[i] * v2[i]);
        }
        return ret;
    }
}

double vectorSum(const std::vector<double>& v1) noexcept
{
    double ret = 0;

    for (auto v : v1)
    {
        ret += v;
    }
    return ret;
}

std::vector<std::vector<double> > vectorTransposeMult(const std::vector<double>& v1, const std::vector<double>& v2) noexcept
{
    std::vector<std::vector<double>> ret;

    for (int i = 0; i < v1.size(); ++i)
    {
        std::vector<double> row;
        for (int j = 0; j < v2.size(); ++j)
        {
            row.push_back(v1[i] * v2[j]);
        }
        ret.push_back(row);
    }
    return ret;

}

std::vector<std::vector<double> > matrixTranspose(const std::vector<std::vector<double>>& m) noexcept
{
    std::vector<std::vector<double>> ret;
    for (int i = 0; i < m[0].size(); ++i)
    {
        std::vector<double> row;
        for (int j = 0; j < m.size(); ++j)
        {
            row.push_back(m[j][i]);
        }
        ret.push_back(row);
    }
    return ret;
}


void normalizeVector(std::vector<double> &v)
{
    double total = 0;
    for (auto vi : v)
    {
        total += vi*vi;
    }
    if (total != 0)
    {
        total = std::sqrt(total);
        for (int i = 0; i < v.size(); ++i)
        {
            if (total == 0 && v[i] == 0)
            {
                std::cout << "Total: " << total << ", V[i]: " << v[i] << std::endl;
                std::cout << "Yup, we got a problem: " << (v[i] / total) << std::endl;
            }
            v[i] = v[i] / total;
        }
    }

}

std::vector<double> averageVectors(const std::vector<std::vector<double>>& vectors) noexcept
{
    std::vector<double> ret;
    for (auto v : vectors)
    {
        double total = 0;
        for (auto value : v)
        {
            total += value;
        }
        ret.push_back(total / vectors.size());
    }
    return ret;
}
