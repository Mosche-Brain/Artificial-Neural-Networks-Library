#include "Input.hpp"

namespace ANN::Models::Layers
{
    Input::Input(int layerSize)
    {
        this->activation = Utils::Activation("linear");

        this->weights = Eigen::MatrixXf::Ones(layerSize, 1);
        this->biases  = Eigen::VectorXf::Zero(layerSize);

        this->_layerSize_   = layerSize;
        this->_layerType_   = LAYER_TYPE::INPUT;
        this->_initialized_ = true;
    }

    Eigen::MatrixXf Input::forward(const Eigen::MatrixXf& input)
    {
        this->outputs = input;
        return input;
    }

    Eigen::MatrixXf Input::backward(const Eigen::MatrixXf& deltaOutput)
    {
        return Eigen::MatrixXf::Zero(_layerSize_, 1);
    }

    void Input::update_weights(float_t rate)
    {

    }

    std::unique_ptr<LayerBase> Input::createUnique(int layerSize)
    {
        return std::make_unique<Input>(layerSize);
    }
}