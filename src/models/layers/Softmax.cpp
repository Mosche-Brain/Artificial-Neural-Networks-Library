/*
 * @author: jaro
 * @name:   Softmax
 * @file:   src/models/layers/Softmax.cpp
 * @date:   02 October 2026 17:18:23
 */

#include <cum/neural_primitives/reductions.hpp>

#include "yann/runtime_config.hpp"

#include "yann/models/layers/Softmax.hpp"

namespace yann::models::layers
{
    Softmax::Softmax(const cum::dim_t lenght, const cum::dim_t axis) : Softmax(cum::Shape{lenght}, axis)
    { }

    Softmax::Softmax(const cum::Shape& shape ,const cum::dim_t axis) : axis_(axis)
    {
        output_shape_ = shape;
    }

    void Softmax::init_parameters(int output_features, int input_features)
    {
        init_parameters(cum::Shape{input_features}, cum::Shape{output_features});
    }

    void Softmax::init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape)
    {
        _initialized_ = true;
    }

    cum::Tensor Softmax::forward(const cum::Tensor& input)
    {
        cache.x = input;

        cum::neural_primitives::softmax(cache.a, cache.x, axis_);

        return cache.a;
    }

    cum::Tensor Softmax::backward(const cum::Tensor& gradient)
    {
        constexpr bool YANN_SOFTMAX_BACKWARD_FUSED_PATH = true;
        constexpr bool YANN_SOFTMAX_BACKWARD_REFERENCE_PATH = true;

        if (runtime_config::fused_kernels())
        {
            if constexpr(YANN_SOFTMAX_BACKWARD_FUSED_PATH)
            {
                throw std::runtime_error("YANN_SOFTMAX_BACKWARD_FUSED_PATH not implemented");
            }
            else
            {
                throw std::runtime_error("YANN_SOFTMAX_BACKWARD_FUSED_PATH weren't compiled");
            }
        }
        else
        {
            if constexpr(YANN_SOFTMAX_BACKWARD_REFERENCE_PATH)
            {
                cum::Tensor grad_output_product_sum = gradient.multiply(cache.da).sum(axis_, true);

                cum::Tensor dL_dX = cache.da * (gradient - grad_output_product_sum);

                return dL_dX;
            }
            else
            {
                throw std::runtime_error("YANN_SOFTMAX_BACKWARD_REFERENCE_PATH weren't compiled");
            }
        }
    }

    void Softmax::collect_parameters(std::vector<Parameter*>& params)
    {

    }

    std::unique_ptr<LayerBase> Softmax::createUnique(const cum::dim_t lenght, const cum::dim_t axis)
    {
        return std::make_unique<Softmax>(lenght, axis);
    }
    std::unique_ptr<LayerBase> Softmax::createUnique(const cum::Shape& shape, cum::dim_t axis)
    {
        return std::make_unique<Softmax>(shape, axis);
    }
} // yann::models::layers