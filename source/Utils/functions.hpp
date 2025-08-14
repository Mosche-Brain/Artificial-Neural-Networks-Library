#pragma once
#include <Eigen/Dense>
#include <limits>
#include <functional>
#include <cmath>

#define Σ(x) x.sum()

double tanh_prim(double x);

double sigmoid(double x);
double sigmoid_prim(double x);

double ReLU(double x);
double ReLU_prim(double x);

double GELU(double x);
double GELU_prim(double x);

double square(double x);
double square_prim(double x);

double linear(double x, double a=1, double b=0);
double linear_prim(double x, double a=1, double b=0);

double pass(double x);
double pass_prim(double x);

double derivative(std::function<double(double)> f, double x);
// double flatten(MatrixXd mat);

//double MSE(double x, double y);

double MSE(Eigen::VectorXd x, Eigen::VectorXd y);

double cross_entropy(Eigen::VectorXd expected, Eigen::VectorXd predicted);

