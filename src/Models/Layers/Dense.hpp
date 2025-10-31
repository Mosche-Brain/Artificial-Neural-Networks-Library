#pragma once

#include "LayerBase.hpp"

namespace ANN::Models::Layers
{
    class Dense : public LayerBase
    {
    public:
        // Dense(int layerSize, int inputWidth, const char* func);
        Dense(int layerSize, const char* func);

        Eigen::MatrixXf forward(const Eigen::MatrixXf& input) override;
        Eigen::MatrixXf backward(const Eigen::MatrixXf& deltaOutput);
        void update_weights(float_t rate);

        static std::unique_ptr<LayerBase> createUnique(int layerSize, const char* func);

        Eigen::VectorXf input;
    protected:
        Eigen::MatrixXf d_weights;
        Eigen::VectorXf d_biases; 
    };
}