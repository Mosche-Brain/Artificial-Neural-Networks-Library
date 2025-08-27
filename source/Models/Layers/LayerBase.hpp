#pragma once

#include <Eigen/Dense>

#include "Utility/Activation.hpp"

namespace SNN::Models::Layers
{
    class LayerBase
    {
    public:
        // LayerBase();

        Eigen::MatrixXf weights;
        Eigen::VectorXf biases;

        Utils::Activation activation;

        virtual Eigen::MatrixXf forward(Eigen::MatrixXf input) = 0;
    };
}   