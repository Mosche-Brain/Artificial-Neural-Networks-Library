
#include <ranges>

#include <cum/functions.hpp>
#include <cum/Core.hpp>

#if defined(ENABLE_DEBUG_OUTPUT)
    #include <iostream>
    #include "utils/formating.hpp"
#endif

#include "yann/runtime_config.hpp"
#include "Input.hpp"

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
        this->input_shape_ = cum::Shape{layerSize, 1};
        this->_layerSize_   = layerSize;
        this->_layerType_   = LayerType::Input;
        this->_initialized_ = true;
    }

    void Input::init_parameters(int output_features, int input_features)
    {

    }

    cum::Tensor Input::forward(const cum::Tensor& input)
    {
        // if () // sprawdza czy batch jest taki sam
        // {
            // cache.resize(cache.x.rows(), cache.z.rows(), input.cols());
        // }
        constexpr bool INPUT_FORWARD_RUNTIME_CHECKS = false;

        if constexpr(INPUT_FORWARD_RUNTIME_CHECKS)
        {
            for (auto [index, dim] : input_shape_ | std::views::enumerate)
            {
                if (dim != input.shape().at(index))
                    throw std::invalid_argument("Input shape mismatch");
            }
        }

        return input;
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