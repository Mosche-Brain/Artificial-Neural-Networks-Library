#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <functional>

#include "Network/Neuron.hpp"

using namespace Eigen;

enum LAYER_FLAG { INPUT, HIDDEN, OUTPUT };

class Layer
{
public:
    Layer(int layer_size, int input_size, std::function<double(double)> func, std::function<double(double)> derivative, bool passive_layer=false);
    Layer(int layer_size, int input_size, const char* func, bool passive_layer=false);

    MatrixXd weights;
    MatrixXd weights_grad;
    VectorXd biases;
    VectorXd biases_grad;
    VectorXd delta;
    VectorXd input;
    VectorXd outputs;
    VectorXd outputs_raw;
    VectorXd derivative_outputs;

    int layer_size;
    int input_size;

    std::function<double(double)> activation_function;
    std::function<double(double)> activation_derivative;

    VectorXd backprop(VectorXd d, double rate);
    VectorXd compute_delta(VectorXd target, bool output_layer=false, const Layer* next=nullptr);
    VectorXd forward(VectorXd input, bool override_output=true);
    void train(MatrixXd data, VectorXd expected, int iter, double rate);

protected:
    LAYER_FLAG layerType;
};
