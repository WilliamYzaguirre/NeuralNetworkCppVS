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

// I'm leaving this function here cause it's funny, but... Will... this is for literally normalizing vectors, not data... Use minmax

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

void minMaxNormalizeVector(std::vector<double>& v, double low, double high)
{
    double max = -100000;
    double min = 1000000;
    for (auto value : v)
    {
        if (value > max)
        {
            max = value;
        }
        if (value < min)
        {
            min = value;
        }
    }
    double divisor = max - min;
    for (int i = 0; i < v.size(); ++i)
    {
        v[i] = (high - low) * ((v[i] - min) / divisor) + low;
    }
}

void minMaxNormalizeVectorAVX2(std::vector<double>& v, double low, double high)
{
    // Need registers to hold max and min. As I go through vector, I will get the 4 biggest and 4 smallest values in each of these
    // At the end I need to compare those 4 values to get actual max and min
    // This is 2(size/4) + 4! compares vs 2(size) compares. 
    // Eg size = 100: 2(100/4) + 24 = 74 vs 2(100) = 200 
    __m256d _max = _mm256_set1_pd(-1000.0);
    __m256d _min = _mm256_set1_pd(1000.0);
    __m256d _mask1, _temp;
    // Iterate through values 4 at a time
    for (int i = 0; i < v.size(); i += 4)
    {
        // Load 4 values
        _temp = _mm256_set_pd(v[i], v[i + 1], v[i + 2], v[i + 3]);

        // Compare to see if any are greater than _max
        _mask1 = _mm256_cmp_pd(_temp, _max, _CMP_GT_OQ);

        // And input(temp) with mask to get new max
        _max = _mm256_and_pd(_temp, _mask1);

        // Compare to see if any are less than _min
        _mask1 = _mm256_cmp_pd(_temp, _min, _CMP_LT_OQ);

        // And input(temp) with mask to get new min
        _min = _mm256_and_pd(_temp, _mask1);
    }

    // Unpack max and min and find the real max and min
    std::vector<double> maxTemp{ _max.m256d_f64[0], _max.m256d_f64[1], _max.m256d_f64[2], _max.m256d_f64[3] };
    std::vector<double> minTemp{ _min.m256d_f64[0], _min.m256d_f64[1], _min.m256d_f64[2], _min.m256d_f64[3] };

    double max = *std::max_element(maxTemp.begin(), maxTemp.end());
    double min = *std::min_element(minTemp.begin(), minTemp.end());

    // If both min and max are zero, then the divisor will be zero, and then we'll divide by zero and get an error.
    // Check if this is the case now, and just assign the vector to all low's
    if (min == 0 && max == 0)
    {
        for (int i = 0; i < v.size(); ++i)
        {
            v[i] = low;
        }
    }
    else if (min == max)
    {
        for (int i = 0; i < v.size(); ++i)
        {
            v[i] = high;
        }
    }
    else
    {
        double divisor = max - min;
        __m256d _div = _mm256_set1_pd(divisor);
        _min = _mm256_set1_pd(min);
        __m256d _mult = _mm256_set1_pd(high - low);
        __m256d _lowRange = _mm256_set1_pd(low);

        for (int i = 0; i < v.size(); i += 4)
        {
            _temp = _mm256_set_pd(v[i], v[i + 1], v[i + 2], v[i + 3]);
            _temp = _mm256_sub_pd(_temp, _min);
            _temp = _mm256_div_pd(_temp, _div);
            _temp = _mm256_fmadd_pd(_mult, _temp, _lowRange);
            v[i] = _temp.m256d_f64[3];
            v[i + 1] = _temp.m256d_f64[2];
            v[i + 2] = _temp.m256d_f64[1];
            v[i + 3] = _temp.m256d_f64[0];
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
