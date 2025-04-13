#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <Eigen/Dense>

#include "utils.hpp"

using namespace Eigen;

class Perceptron 
{
public:
    Perceptron(int32_t inputs);
    void activate();
    void train(Matrix<double, Dynamic, Dynamic> x_train, VectorXd expected, int n_iter, double learning_rate);
    //void train(int x_train, std::vector<double> expected, int n_iter, double learning_rate);
    double forward(VectorXd x);
    double activation_function(double x);
    std::vector<double> predict(MatrixXd x);
    void forward_signal(std::vector<double> x);


    double bias;
    VectorXd weights;
protected:
    //std::vector<Perceptron*> connections;    
};