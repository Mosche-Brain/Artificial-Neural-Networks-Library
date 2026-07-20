#include "Input.hpp"

#include <cum/functions.hpp>
#include <cum/Core.hpp>

#if defined(ENABLE_DEBUG_OUTPUT)
    #include <iostream>
    #include "utils/logs.hpp"
#endif

namespace yann::models::layers
{
    Input::Input(int layerSize)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        // if(runtime_config::DEBUG_VEBOSITY >= 3)
            std::cout << "\t\t\t" << "Initializing Input layer with " << layerSize << " neurons...\n";
        #endif
        // this->activation = utils::Activation("linear");
        #if defined(ENABLE_DEBUG_OUTPUT)
            std::cout << "Initializing activation function: linear\n";
        #endif
        cum::functions::getFunctionByName(&this->activation, "linear");
        #if defined(ENABLE_DEBUG_OUTPUT)
            std::cout << "Activation function initialized: " << '\n';
        #endif
        // this->weights = cum::Matrix(2, 1, 1);
        this->weights = Parameter::Ones(layerSize, 1);
        #if defined(ENABLE_DEBUG_OUTPUT)
            std::cout << "Weights initialized with dimensions: " << '\n';
        #endif
        this->biases  = Parameter::Zeros(layerSize, 1);
        #if defined(ENABLE_DEBUG_OUTPUT)
            std::cout << "Biases initialized with dimensions: " << '\n';
        #endif
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