//
// Created on 2018/04/14 at 12:58.
//

#ifndef PA1_UTILS_HPP
#define PA1_UTILS_HPP

#include "Eigen/Eigen"

class Utils
{
public:
    static Eigen::MatrixXd Hilbert(int size);
    static double Cond2(const Eigen::MatrixXd& matrix);
};

Eigen::MatrixXd Utils::Hilbert(const int size)
{
    Eigen::MatrixXd ret(size, size);
    for (auto i = 0; i < size; i++)
        for (auto j = 0; j < size; j++)
            ret(i, j) = 1.0 / (i + j + 1);
    return ret;
}

double Utils::Cond2(const Eigen::MatrixXd& matrix)
{
    auto norm2 = matrix.lpNorm<2>();
    auto invNorm2 = matrix.inverse().lpNorm<2>();
    return norm2 * invNorm2;
}

#endif //PA1_UTILS_HPP
