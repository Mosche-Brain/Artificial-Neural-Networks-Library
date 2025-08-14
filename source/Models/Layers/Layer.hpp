#pragma once
#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <functional>
#include <map>

#include "Models/Neuron.hpp"
#include "Utils/activation_map.hpp"
#include "Utils/functions.hpp"

using namespace Eigen;

class Layer
{
public:
    Layer(int layer_size, int input_size, std::function<double(double)> func, std::function<double(double)> derivative, bool passive_layer=false);
    Layer(int layer_size, int input_size, const char* func, bool passive_layer=false);
    ~Layer() = default;

    MatrixXd weights;
    VectorXd biases;
    VectorXd delta;
    VectorXd outputs;
    VectorXd derivative_outputs;
    
    int size() const;
    int inputWidth() const;
    
    std::function<double(double)> activation_function;
    std::function<double(double)> activation_derivative;
    
    virtual VectorXd forward(VectorXd x, bool derivatives=false);
    virtual void train(MatrixXd data, VectorXd expected, int iter, double rate);
    
protected:
    VectorXd outputs_raw;
    int layer_size;
    int input_size;
};
