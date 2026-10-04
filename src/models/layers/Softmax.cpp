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
        _layerType_ = LayerType::Softmax;
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

        YANN_LOG(3, "Y = softmax(X) over {} axis |", axis_);
        // cum::neural_primitives::softmax(cache.a, cache.x, axis_);
        cache.a = cache.x.softmax(axis_);
        YANN_LOG(3, "pońćzochy", "");
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
                YANN_LOG(3, "(dL_dY * Y).sum(axis={})", axis_);
                cum::Tensor grad_output_product_sum = gradient.multiply(cache.a).sum(axis_, true);

                cache.da = cum::Tensor(cache.a.shape(), cache.a.type(), cache.a.format());

                YANN_LOG(3, "dL_dX = dL_dY (dL_dY - grad_outpu_product_sum)", axis_);
                cum::Tensor dL_dX = cache.a.multiply(gradient - grad_output_product_sum);

                YANN_LOG(3, "zabezpieczenie", 0);
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