#include "Utils.hpp"
#include "LinearSolve.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;

void GetHilbertCondTable(int n, ostream &out);

int main(int argc, const char *argv[])
{
    string type = argv[1];
    auto size = stoi(argv[2]);
    auto filePath = argv[3];
    ofstream fout(filePath, ios::ate);
    vector<Eigen::MatrixXd> retVec;
    auto mat = Utils::Hilbert(size);
    auto res = mat * Eigen::VectorXd::Ones(size);

    if (type == "Hilbert")
    {
        GetHilbertCondTable(size, fout);
    }
    else if (type == "Gauss")
    {
        auto ans = LinearSolve::GaussSolve(mat, res);
        fout << ans;
    }
    else if (type == "Jacobi")
    {
        auto history = LinearSolve::JacobiSolve(mat, res, Eigen::VectorXd::Zero(size), stoi(argv[4]));
        for (const auto &item : history)
            fout << item << endl
                 << endl
                 << endl;
    }
    else if (type == "GaussSeidel")
    {
        auto history = LinearSolve::GaussSeidelSolve(mat, res, Eigen::VectorXd::Zero(size), stoi(argv[4]));
        for (const auto &item : history)
            fout << item << endl
                 << endl
                 << endl;
    }
    else if (type == "SOR")
    {
        auto history = LinearSolve::SorSolve(mat, res, Eigen::VectorXd::Zero(size), stoi(argv[4]), stod(argv[5]));
        for (const auto &item : history)
            fout << item << endl
                 << endl
                 << endl;
    }
    else if (type == "PCG")
    {
        auto history = LinearSolve::PcgSolve(mat, res, Eigen::VectorXd::Zero(size), stoi(argv[4]));
        for (const auto &item : history)
            fout << item << endl
                 << endl
                 << endl;
    }
    else
    {
        cout << "Unkown command!\n";
    }

    fout.close();

    return 0;
}

void GetHilbertCondTable(int n, ostream &out)
{
    for (auto i = 1; i < n; i++)
        out << i << ' ' << Utils::Cond2(Utils::Hilbert(i)) << endl;
}