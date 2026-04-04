#include "Input.hpp"

namespace YANN::Models::Layers
{
    Input::Input(int layerSize)
    {
        // this->activation = Utils::Activation("linear");
        this->activation = Utils::getActivationByName("linear");

        this->weights = matrix_t::Ones(layerSize, 1);
        this->biases  = vector_t::Zero(layerSize);

        this->_layerSize_   = layerSize;
        this->_layerType_   = LAYER_TYPE::INPUT;
        this->_initialized_ = true;
    }

    matrix_t Input::forward(const matrix_t& input)
    {
        this->outputs = input;
        return input;
    }

    matrix_t Input::backward(const matrix_t& deltaOutput)
    {
        // return matrix_t::Zero(_layerSize_, 1);
        return math_api::createMatrix(_layerSize_, 1, static_cast<numeric_t>(1));
    }

    void Input::update_weights(numeric_t rate)
    {

    }

    std::unique_ptr<LayerBase> Input::createUnique(int layerSize)
    {
        return std::make_unique<Input>(layerSize);
    }
}