#include "Dense.hpp"

namespace SNN::Models::Layers
{
    // Dense::Dense(int layerSize, int inputWidth, const char* func) : LayerBase()
    Dense::Dense(int layerSize, const char* func)
    {
        activation  = Utils::Activation(func);
        _layerSize_ = layerSize;
        // if(inputWidth > 0)
        // {
        //     initParameters(layerSize, inputWidth);
        // }
    }
    
    Eigen::MatrixXf Dense::forward(const Eigen::MatrixXf& input)
    {
        Eigen::MatrixXf result = activation.matrixFunction((weights * input) + biases);
        outputs = result;

        return result;
    }

    std::unique_ptr<LayerBase> Dense::getUnique()
    {
        return std::make_unique<Dense>(*this);
    }
}