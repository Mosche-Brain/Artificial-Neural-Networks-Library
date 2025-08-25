#pragma once

#include <Eigen/Dense>

namespace SNN::Models::Layers
{
    class LayerBase
    {
    public:
        Eigen::MatrixXf weights;
        Eigen::VectorXf biases;

        virtual void forward(Eigen::MatrixXf input);
    };
}