#pragma once

#include "LayerBase.hpp"

namespace SNN::Models::Layers
{
    class Input : LayerBase
    {
    public:
        Input(int layerSize);

        Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
        
        std::unique_ptr<LayerBase> getUnique() override;
    };
}