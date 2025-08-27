#pragma once

#include "LayerBase.hpp"

namespace SNN::Models::Layers
{
    class Dense : public LayerBase
    {
    public:
        Dense(int layerSize, int inputWidth, const char* func);

        Eigen::MatrixXf forward(Eigen::MatrixXf input) override;
    };
}