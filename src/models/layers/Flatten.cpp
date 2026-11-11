 //
// Created by jaro on 7/24/26.
//

#include "yann/models/layers/Flatten.hpp"

#include "yann/runtime_config.hpp"

/*
 * Trzeba rozwiązać zachowanie wymiaru batcha przy spłaszczaniu próbek
 */

namespace yann::models::layers
{
    Flatten::Flatten()
    {
        this->_kind = Kind::Flatten;
    }

    Flatten::Flatten(const cum::Shape& shape)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        YANN_LOG(1, "Initializing flatten layer with input shape", "");
        #endif

        init_parameters(shape, {});
        this->_kind = Kind::Flatten;
    }

    void Flatten::init_parameters(int output_features, int input_features)
    {
        init_parameters(cum::Shape{input_features}, cum::Shape{output_features});
    }

    void Flatten::init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape)
    {
        this->input_shape_ = input_shape;
        this->shape_ = input_shape;
        if (!output_shape.empty())
        {
            this->output_shape_ = output_shape;
        }
        else
        {
            cum::dim_t total = 1;
            for (auto d : input_shape)
                total *= d;
            this->output_shape_ = cum::Shape{total};
        }
        this->_initialized = true;
    }

    cum::Tensor Flatten::forward(const cum::Tensor& input)
    {
        return input.reshape(output_shape_);
    }

    cum::Tensor Flatten::backward(const cum::Tensor& deltaOutput)
    {
        return deltaOutput.reshape(input_shape_);
    }

    std::unique_ptr<LayerBase> Flatten::createUnique()
    {
        return std::make_unique<Flatten>();
    }

    std::unique_ptr<LayerBase> Flatten::createUnique(const cum::Shape& shape)
    {
        return std::make_unique<Flatten>(shape);
    }
} // yann::models::layers