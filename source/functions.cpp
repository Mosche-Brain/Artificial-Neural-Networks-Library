#include "functions.hpp"

double tanh_prim(double x)
{
    return 1 - std::pow(tanh(x), 2);
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_prim(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

double RELu(double x)
{
    return x > 0 ? x : 0;
}

double RELu_prim(double x)
{
    return x > 0 ? 1 : 0;
}

double square(double x)
{
    return x * x;
}

double linear(double x, double a, double b)
{
    return a * x + b;
}

double linear_prim(double x, double a, double b)
{
    return a;
}

double pass(double x)
{
    return x;
}

double pass_prim(double x)
{
    return 1;
}

//double MSE(double x, double y)


double MSE(Eigen::VectorXd expected, Eigen::VectorXd predicted)
{
    Eigen::VectorXd delta = expected - predicted;

    return delta.squaredNorm() / predicted.size();
}

double cross_entropy(Eigen::VectorXd expected, Eigen::VectorXd predicted)
{
    double loss = 0.0;
    for (int i = 0; i < predicted.size(); ++i) {
        loss -= expected[i] * std::log(predicted[i]) + (1 - expected[i]) * std::log(1 - predicted[i]);
    }
    return loss / predicted.size();
}
