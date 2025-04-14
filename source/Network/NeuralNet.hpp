#pragma once
#include "Network/Layer.hpp"
#include "functions.hpp"

#include <Eigen/Dense>

#include <nlohmann/json.hpp>

using namespace Eigen;

using json = nlohmann::json;

class NeuralNet
{
public:
    NeuralNet(int layers_n, VectorXi layers_size);
    NeuralNet(std::vector<Layer*> _layers_);


    void addLayer(Layer* layer);
    void train(MatrixXd train_data, VectorXd expected, uint n_iter, float rate);
    void backpropagate(VectorXd expected, double rate);
    void update_weights(float rate);
    void feedforward();

    double transfer_derivative(double x);

    VectorXd forward(MatrixXd input);
    VectorXd forward(VectorXd input);

    Layer* inputLayer();
    Layer* outputLayer();
    
    json export_to_json();


    std::vector<Layer*> layers;
protected:
    int layer_size;

    VectorXd forward(int currentIndex, int nextIndex);
};
