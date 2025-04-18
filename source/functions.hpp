#pragma once
#include <Eigen/Dense>
#include <cmath>

#define Σ(x) x.sum()

double tanh_prim(double x);

double sigmoid(double x);
double sigmoid_prim(double x);

double RELu(double x);
double RELu_prim(double x);

double square(double x);
double square_prim(double x);

double linear(double x, double a=1, double b=0);
double linear_prim(double x, double a=1, double b=0);

double pass(double x);
double pass_prim(double x);

double MSE(double x, double y);

double MSE(Eigen::VectorXd x, Eigen::VectorXd y);