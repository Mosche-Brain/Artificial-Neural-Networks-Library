#include <Eigen/Dense>
#include <iostream>
#include <algorithm>
#include <functional>

#include "Network/Neuron.hpp"

using namespace Eigen;

class Layer
{
public:
    Layer(int layer_size, int input_size, std::function<double(double)> func, bool passive_layer=false);
    MatrixXd weights;
    VectorXd biases;
    VectorXd outputs;
    VectorXd delta;

    int layer_size;
    int input_size;

    std::function<double(double)> activation_function;

    VectorXd forward(MatrixXd input, bool override_output=true);
    void train(MatrixXd data, VectorXd expected, int iter, double rate);

    VirtualNeuron neuron(int index);
    std::vector<VirtualNeuron> getLayer();
};
