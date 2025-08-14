#pragma once
#include "Models/Layers/Layer.hpp"
#include "Models/Layers/Layers.hpp"
#include "Utils/functions.hpp"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <fstream>
#include <memory>
#include <utility>
#include <print>

using json = nlohmann::json;

class NeuralNet
{
public:
    NeuralNet(int layers_n, VectorXi layers_size);
    NeuralNet(std::vector<std::unique_ptr<Layer>> _layers_) ;
    NeuralNet(std::vector<Layers::INIT_PARAMS> topology);

    void setLossFunction(std::function<double(VectorXd, VectorXd)> func);
    void addLayer(Layer* layer);
    void train(MatrixXd train_data, VectorXd expected, uint n_iter, float rate, bool print_output=false);
    void backpropagate(VectorXd expected);
    void update_weights(float rate);
    void feedforward();

    VectorXd forward(MatrixXd input, bool derivatives=false);
    MatrixXd predict(MatrixXd input);

    void export_to_json(const char* filename);
    void load_from_json(const char* filename);

    VectorXd input;
    // std::vector<Layer*> layers;
    std::vector<std::unique_ptr<Layer>> layers;
protected:
    std::function<double(VectorXd, VectorXd)> loss_function;
};
