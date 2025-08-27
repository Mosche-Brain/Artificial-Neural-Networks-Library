#include "Dense.hpp"

namespace SNN::Models::Layers
{
    // Dense::Dense(int layerSize, int inputWidth, const char* func) : LayerBase()
    Dense::Dense(int layerSize, int inputWidth, const char* func)
    {
        activation = Utils::Activation(func);

        weights = Eigen::MatrixXf::Random(layerSize, inputWidth);
        biases  = Eigen::VectorXf::Random(layerSize);
    }

    Eigen::MatrixXf Dense::forward(Eigen::MatrixXf input)
    {
        return activation.matrixFunction((weights * input) + biases);
    }
}