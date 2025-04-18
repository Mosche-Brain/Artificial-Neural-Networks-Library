#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <functional>

#include "Network/Neuron.hpp"

using namespace Eigen;

using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXd;

class Layer
{
public:
    Layer(int layer_size, int input_size, std::function<double(double)> func, std::function<double(double)> derivative, bool passive_layer=false);
    Layer(int layer_size, int input_size, const char* func, bool passive_layer=false);

    MatrixXd weights;
    VectorXd biases;
    VectorXd delta;
    VectorXd outputs;
    VectorXd derivative_outputs;

    int layer_size;
    int input_size;

    std::function<double(double)> activation_function;
    std::function<double(double)> activation_derivative;

    VectorXd compute_delta(VectorXd target, bool output_layer=false);
    VectorXd forward(MatrixXd input, bool override_output=true);
    void train(MatrixXd data, VectorXd expected, int iter, double rate);

};
