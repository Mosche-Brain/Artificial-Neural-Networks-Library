#pragma once

#include <Eigen/Dense>

#include <initializer_list>
#include <memory>
#include <vector>

#include "Layers/Layers.hpp"
#include "Utility/loss.hpp"

namespace ANN::Models
{
    using Topology = std::vector<std::unique_ptr<Layers::LayerBase>>;
    class Sequential
    {
    public:
        Sequential() ;
        // Sequential(std::vector<std::unique_ptr<Layers::LayerBase>> newTopology);
        Sequential(std::initializer_list<std::unique_ptr<Layers::LayerBase>> newTopology);

        void addLayer(std::unique_ptr<Layers::LayerBase> layer);

        Eigen::MatrixXf forward(Eigen::MatrixXf input);
        void backward(const Eigen::MatrixXf& d_output);

        void fit(const Eigen::MatrixXf& X, const Eigen::MatrixXf& Y, float_t rate, int epochs);
        void updateParams(float_t rate);

        Eigen::MatrixXf getWeights(int layer_idx);
        
        std::vector<std::unique_ptr<Layers::LayerBase>> topology;
    protected:
        Utils::loss::LossFunction loss_function = Utils::loss::LossFunction::mse;
    };
}