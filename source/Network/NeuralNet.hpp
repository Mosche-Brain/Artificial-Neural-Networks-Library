#pragma once
#include "Network/Layer.hpp"
#include "functions.hpp"

#include <Eigen/Dense>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

class NeuralNet
{
public:
    NeuralNet(int layers_n, VectorXi layers_size);
    NeuralNet(std::vector<Layer*> _layers_);


    void setLossFunction(std::function<double(VectorXd, VectorXd)> func);
    void addLayer(Layer* layer);
    void train(MatrixXd train_data, MatrixXd expected, uint n_iter, float rate);
    void backpropagate(VectorXd expected);
    void update_weights(float rate);
    void feedforward();

    VectorXd forward(MatrixXd input, bool derivatives=false);
    MatrixXd predict(MatrixXd input);

    json export_to_json();

    VectorXd input;
    std::vector<Layer*> layers;
protected:
    std::function<double(VectorXd, VectorXd)> loss_function;

    

    int layer_size;
};
