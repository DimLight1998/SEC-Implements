//
// Created on 2018/04/14 at 13:29.
//

#ifndef PA1_LINEARSOLVE_HPP
#define PA1_LINEARSOLVE_HPP

#include <algorithm>
#include <vector>
#include "Eigen/Eigen"

class LinearSolve
{
  public:
    static Eigen::VectorXd GaussSolve(
        const Eigen::MatrixXd &a, const Eigen::VectorXd &b);

    static std::vector<Eigen::VectorXd> JacobiSolve(
        const Eigen::MatrixXd &a, const Eigen::VectorXd &b,
        const Eigen::VectorXd &start, int maxIter);

    static std::vector<Eigen::VectorXd> GaussSeidelSolve(
        const Eigen::MatrixXd &a, const Eigen::VectorXd &b,
        const Eigen::VectorXd &start, int maxIter);

    static std::vector<Eigen::VectorXd> SorSolve(
        const Eigen::MatrixXd &a, const Eigen::VectorXd &b,
        const Eigen::VectorXd &start, int maxIter, double omega = 1.0);

    static std::vector<Eigen::VectorXd> PcgSolve(
        const Eigen::MatrixXd &a, const Eigen::VectorXd &b,
        const Eigen::VectorXd &start, int maxIter);

    static Eigen::VectorXd LowerTriangleSolve(
        const Eigen::MatrixXd &a, const Eigen::VectorXd &b);

    static Eigen::VectorXd UpperTriangleSolve(
        const Eigen::MatrixXd &a, const Eigen::VectorXd &b);

    static Eigen::VectorXd LowerUpperSolve(
        const Eigen::MatrixXd &lower, const Eigen::MatrixXd &upper, const Eigen::VectorXd &b);

  private:
    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd> GetRowBalancedMatrix(const Eigen::MatrixXd &matrix);
    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd> DoolittleDecomposition(const Eigen::MatrixXd &matrix);
};

Eigen::VectorXd LinearSolve::GaussSolve(
    const Eigen::MatrixXd &a, const Eigen::VectorXd &b)
{
    auto balanced = GetRowBalancedMatrix(a);
    auto newA = std::move(balanced.second);
    auto newB = b;
    for (auto i = 0; i < b.rows(); i++)
        newB[i] /= balanced.first(i, i);
    auto decomposition = DoolittleDecomposition(newA);
    auto l = std::move(decomposition.first);
    auto u = std::move(decomposition.second);

    return LowerUpperSolve(l, u, newB);
}

std::vector<Eigen::VectorXd> LinearSolve::JacobiSolve(
    const Eigen::MatrixXd &a, const Eigen::VectorXd &b, const Eigen::VectorXd &start, const int maxIter)
{
    auto ans1 = start;
    auto ans2 = start;
    auto newer = &ans1;
    auto elder = &ans2;
    std::vector<Eigen::VectorXd> ret;
    ret.reserve(maxIter);

    for (auto i = 0; i < maxIter; i++)
    {
        for (auto row = 0; row < b.rows(); row++)
        {
            (*newer)[row] = b[row];
            for (auto col = 0; col < row; col++)
                (*newer)[row] -= a(row, col) * (*elder)[col];
            for (auto col = row + 1; col < a.cols(); col++)
                (*newer)[row] -= a(row, col) * (*elder)[col];
            (*newer)[row] /= a(row, row);
        }

        ret.push_back(*newer);

        std::swap(elder, newer);
    }

    return ret;
}

std::vector<Eigen::VectorXd> LinearSolve::GaussSeidelSolve(
    const Eigen::MatrixXd &a, const Eigen::VectorXd &b, const Eigen::VectorXd &start, const int maxIter)
{
    auto ans = start;
    std::vector<Eigen::VectorXd> ret;
    ret.reserve(maxIter);

    for (auto i = 0; i < maxIter; i++)
    {
        for (auto row = 0; row < b.rows(); row++)
        {
            ans[row] = b[row];
            for (auto col = 0; col < row; col++)
                ans[row] -= a(row, col) * ans[col];
            for (auto col = row + 1; col < a.cols(); col++)
                ans[row] -= a(row, col) * ans[col];
            ans[row] /= a(row, row);
        }

        ret.push_back(ans);
    }

    return ret;
}

std::vector<Eigen::VectorXd> LinearSolve::SorSolve(
    const Eigen::MatrixXd &a, const Eigen::VectorXd &b, const Eigen::VectorXd &start, const int maxIter, double omega)
{
    auto iter = start;
    std::vector<Eigen::VectorXd> ret;
    ret.reserve(maxIter);

    for (auto i = 0; i < maxIter; i++)
    {
        for (auto row = 0; row < b.rows(); row++)
        {
            auto oldVal = iter[row];
            iter[row] = b[row];
            for (auto col = 0; col < row; col++)
                iter[row] -= a(row, col) * iter[col];
            for (auto col = row + 1; col < a.cols(); col++)
                iter[row] -= a(row, col) * iter[col];
            iter[row] *= (omega / a(row, row));
            iter[row] += (1 - omega) * oldVal;
        }
        ret.push_back(iter);
    }

    return ret;
}

std::vector<Eigen::VectorXd> LinearSolve::PcgSolve(
    const Eigen::MatrixXd &a, const Eigen::VectorXd &b, const Eigen::VectorXd &start, const int maxIter)
{
    auto size = a.rows();
    std::vector<Eigen::VectorXd> ret;
    ret.reserve(maxIter);

    // Use the matrix used in SSOR method.
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(size, size);
    u.bottomLeftCorner(size, size) = a.bottomLeftCorner(size, size);
    auto l = u;
    u.transposeInPlace();

    for (auto row = 0; row < size; row++)
        for (auto col = row; col < size; col++)
            u(row, col) /= a(row, row);

    // Initialization.
    Eigen::VectorXd lastR = b - a * start;
    Eigen::VectorXd newR;
    Eigen::VectorXd lastZ = LowerUpperSolve(l, u, lastR);
    Eigen::VectorXd newZ;
    auto p = lastZ;
    auto ans = start;

    // Iteration.
    for (auto i = 0; i < maxIter; i++)
    {
        auto alpha = (lastZ.dot(lastR)) / (p.dot((a * p)));
        ans += alpha * p;
        newR = lastR - alpha * (a * p);
        newZ = LowerUpperSolve(l, u, newR);
        auto beta = (newZ.dot(newR)) / (lastZ.dot(lastR));
        p = newZ + beta * p;

        lastR = newR;
        lastZ = newZ;

        ret.push_back(ans);
    }

    return ret;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LinearSolve::GetRowBalancedMatrix(const Eigen::MatrixXd &matrix)
{
    Eigen::MatrixXd ret1 = Eigen::MatrixXd::Zero(matrix.rows(), matrix.rows());
    auto ret2 = matrix;

    for (auto i = 0; i < matrix.rows(); i++)
    {
        auto max = 0.0;
        for (auto j = 0; j < matrix.cols(); j++)
            if (std::abs(matrix(i, j)) > max)
                max = std::abs(matrix(i, j));
        ret1(i, i) = max;
        for (auto j = 0; j < matrix.cols(); j++)
            ret2(i, j) /= max;
    }

    return std::make_pair(ret1, ret2);
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> LinearSolve::DoolittleDecomposition(const Eigen::MatrixXd &matrix)
{
    auto size = matrix.rows();
    Eigen::MatrixXd retL, retU;
    retL = Eigen::MatrixXd::Identity(size, size);
    retU = Eigen::MatrixXd::Zero(size, size);

    for (auto i = 0; i < size - 1; i++)
    {
        // Compute row i of retU.
        for (auto j = i; j < size; j++)
        {
            retU(i, j) = matrix(i, j);
            for (auto k = 0; k < i; k++)
                retU(i, j) -= retL(i, k) * retU(k, j);
        }

        // Compute col i of retL.
        for (auto j = i + 1; j < size; j++)
        {
            retL(j, i) = matrix(j, i);
            for (auto k = 0; k < i; k++)
                retL(j, i) -= retL(j, k) * retU(k, i);
            retL(j, i) /= retU(i, i);
        }
    }

    retU(size - 1, size - 1) = matrix(size - 1, size - 1);
    for (auto i = 0; i < size - 1; i++)
        retU(size - 1, size - 1) -= retL(size - 1, i) * retU(i, size - 1);

    return std::make_pair(retL, retU);
}

Eigen::VectorXd LinearSolve::LowerTriangleSolve(
    const Eigen::MatrixXd &a, const Eigen::VectorXd &b)
{
    auto ret = b;

    for (auto i = 0; i < ret.rows(); i++)
    {
        for (auto j = 0; j < i; j++)
            ret[i] -= a(i, j) * ret[j];
        ret[i] /= a(i, i);
    }

    return ret;
}

Eigen::VectorXd LinearSolve::UpperTriangleSolve(
    const Eigen::MatrixXd &a, const Eigen::VectorXd &b)
{
    auto ret = b;

    for (auto i = ret.rows() - 1; i >= 0; i--)
    {
        for (auto j = i + 1; j < ret.rows(); j++)
            ret[i] -= a(i, j) * ret[j];
        ret[i] /= a(i, i);
    }

    return ret;
}

Eigen::VectorXd LinearSolve::LowerUpperSolve(
    const Eigen::MatrixXd &lower, const Eigen::MatrixXd &upper, const Eigen::VectorXd &b)
{
    auto y = LowerTriangleSolve(lower, b);
    auto ret = UpperTriangleSolve(upper, y);
    return ret;
}

#endif //PA1_LINEARSOLVE_HPP
