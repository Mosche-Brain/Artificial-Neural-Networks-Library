#include "Input.hpp"

namespace SNN::Models::Layers
{
    Input::Input(int layerSize)
    {
        this->activation = Utils::Activation("linear");

        this->weights = Eigen::MatrixXf::Ones(layerSize, 1);
        this->biases  = Eigen::VectorXf::Zero(layerSize);
    }

    Eigen::MatrixXf Input::forward(const Eigen::MatrixXf& input)
    {
        this->outputs = input;
        return input;
    }

    std::unique_ptr<LayerBase> Input::getUnique()
    {
        // return std::make_unique<Input>(*this);
    }
}