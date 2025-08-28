#pragma once

#include <Eigen/Dense>

#include <memory>
#include <vector>

#include "Layers/Layers.hpp"

namespace SNN::Models
{
    class Sequential
    {
    public:
        Sequential();
        Sequential(std::vector<std::unique_ptr<Layers::LayerBase>> topology);

        void addLayer(std::unique_ptr<Layers::LayerBase> layer);

        Eigen::MatrixXf forward(Eigen::MatrixXf input);
    
    protected:
        std::vector<std::unique_ptr<Layers::LayerBase>> topology;
    };
    
}