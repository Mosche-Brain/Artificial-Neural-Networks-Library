#include "Input.hpp"

#include <cum/functions.hpp>
#include <cum/Core.hpp>

#if defined(ENABLE_DEBUG_OUTPUT)
    #include <iostream>
    #include "utils/formating.hpp"
#endif

#include "yann/runtime_config.hpp"

namespace yann::models::layers
{
    Input::Input(int layerSize)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        YANN_LOG(3, "Initializing Input layer with {} neurons...", layerSize);
        #endif

        // cum::functions::get_function_by_name(&this->activation, "linear");

        // this->weights = cum::Matrix(2, 1, 1);
        // this->weights = Parameter::Ones(layerSize, 1);

        // this->biases  = Parameter::Zeros(layerSize, 1);

        this->_layerSize_   = layerSize;
        this->_layerType_   = LAYER_TYPE::INPUT;
        this->_initialized_ = true;
    }

    cum::Matrix Input::forward(const cum::Matrix& input)
    {
        this->outputs = input;
        return this->inputs;
    }

    cum::Matrix Input::backward(const cum::Matrix& deltaOutput)
    {
        return deltaOutput;
        // return cum::Matrix::Zeros(_layerSize_, 1);
        // return cum::Matrix(_layerSize_, 1, 1._c);
        // return cum::Matrix(_layerSize_, 1, 1._c);
    }

    void Input::update_weights(cum::cumeric_t rate)
    {

    }

    void Input::collect_parameters(std::vector<Parameter*>& params)
    {

    }

    std::unique_ptr<LayerBase> Input::createUnique(int layerSize)
    {
        return std::make_unique<Input>(layerSize);
    }
}