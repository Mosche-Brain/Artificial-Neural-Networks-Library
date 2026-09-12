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

    void Input::initParameters(int output_features, int input_features)
    {

    }

    cum::Tensor Input::forward(const cum::Tensor& input)
    {
        // if (input.cols() != cache.x.cols()) // sprawdza czy batch jest taki sam
        // {
        //     cache.resize(cache.x.rows(), cache.z.rows(), input.cols());
        // }
        this->cache.x = input;
        this->cache.a = input;
        return this->cache.a;
    }

    cum::Tensor Input::backward(const cum::Tensor& deltaOutput)
    {
        return deltaOutput;
        // return cum::Matrix::Zeros(_layerSize_, 1);
        // return cum::Matrix(_layerSize_, 1, 1._c);
        // return cum::Matrix(_layerSize_, 1, 1._c);
    }

    void Input::collect_parameters(std::vector<Parameter*>& params)
    {

    }

    std::unique_ptr<LayerBase> Input::createUnique(int layerSize)
    {
        return std::make_unique<Input>(layerSize);
    }
}