#include "Input.hpp"

#include <cum/functions.hpp>

namespace YANN::Models::Layers
{
    Input::Input(int layerSize)
    {
        // this->activation = Utils::Activation("linear");
        cum::functions::getFunctionByName(&this->activation, "linear");

        this->weights = cum::Matrix(layerSize, 1, 1);
        this->biases  = cum::Matrix(layerSize, 1, 0);

        this->_layerSize_   = layerSize;
        this->_layerType_   = LAYER_TYPE::INPUT;
        this->_initialized_ = true;
    }

    cum::Matrix Input::forward(const cum::Matrix& input)
    {
        this->outputs = input;
        return input;
    }

    cum::Matrix Input::backward(const cum::Matrix& deltaOutput)
    {
        // return cum::Matrix::Zero(_layerSize_, 1);
        return cum::Matrix(_layerSize_, 1, static_cast<cum::cumeric_t>(1));
    }

    void Input::update_weights(cum::cumeric_t rate)
    {

    }

    std::unique_ptr<LayerBase> Input::createUnique(int layerSize)
    {
        return std::make_unique<Input>(layerSize);
    }
}