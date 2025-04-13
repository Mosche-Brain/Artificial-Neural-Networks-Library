#include "functions.hpp"

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double RELu(double x)
{
    return x > 0 ? x : 0;
}

double square(double x)
{
    return x * x;
}

double linear(double x, double a, double b)
{
    return a * x + b;
}

double linear_standard(double x)
{
    return x;
}