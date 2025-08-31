#pragma once

#include "LayerBase.hpp"

namespace SNN::Models::Layers
{
    class Input : public LayerBase
    {
    public:
        Input(int layerSize);

        Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
        Eigen::MatrixXf backward(const Eigen::MatrixXf& deltaOutput);
        void update_weights(float_t rate);

        static std::unique_ptr<LayerBase> createUnique(int layerSize);
    };
}