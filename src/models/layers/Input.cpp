
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
    Input::Input(int layerSize) : Input(cum::Shape{layerSize})
    {
    }

    Input::Input(const cum::Shape& shape)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        YANN_LOG(3, "Initializing Input layer with shape...", "");
        #endif

        this->input_shape_ = shape;
        this->output_shape_ = shape;
        this->_kind   = Kind::Input;
        this->_initialized = true;
    }

    void Input::init_parameters(int output_features, int input_features)
    {
        this->input_shape_ = cum::Shape{input_features};
        this->output_shape_ = cum::Shape{output_features};
        this->_initialized = true;
    }

    void Input::init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape)
    {
        this->input_shape_ = input_shape;
        this->output_shape_ = output_shape.empty() ? input_shape : output_shape;
        this->_initialized = true;
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
    }

    void Input::collect_parameters(std::vector<Parameter*>& params)
    {

    }

    std::unique_ptr<LayerBase> Input::createUnique(int layerSize)
    {
        return std::make_unique<Input>(layerSize);
    }

    std::unique_ptr<LayerBase> Input::createUnique(const cum::Shape& shape)
    {
        return std::make_unique<Input>(shape);
    }
}