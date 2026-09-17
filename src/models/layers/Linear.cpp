//
// Created by jaro on 8/31/26.
//

#include "yann/logging/Logger.hpp"
#include "yann/runtime_config.hpp"

#include "yann/models/layers/Linear.hpp"

#define ENABLE_RUNTIME_CHECKS true

/*
 * This layer should not allocate both cache.a and cache.z
 */

namespace yann::models::layers
{
    Linear::Linear(const int layerSize)
    {
        YANN_LOG(1, "Initializing Linear layer with {} neurons...", layerSize);
        _layerSize_ = layerSize;

        this->_layerType_ = LayerType::Linear;
    }

    void Linear::init_parameters(int output_features, int input_features)
    {
        this->weights_      = Parameter::Uniform(output_features, input_features);   /* neurons * input_length */
        this->biases_       = Parameter::Zeros(output_features, 1);               /* Column-Vector */
        this->cache.x       = cum::Tensor({input_features, 1}, cum::default_type, cum::layout::IO);   /* Column-Vector */
        this->cache.a       = cum::Tensor({output_features, 1}, cum::default_type, cum::layout::IO);
        this->cache.dz      = cum::Tensor::Ones({output_features, 1}, cum::default_type, cum::layout::IO);
        this->_initialized_ = true;
    }

    cum::Tensor Linear::forward(const cum::Tensor& input)
    {
        constexpr bool YANN_LINEAR_FORWARD_FUSED_PATH = true;
        constexpr bool YANN_LINEAR_FORWARD_REFERENCE_PATH = true;

        this->cache.x = input;

        if(runtime_config::fused_kernels())
        {
            if constexpr(YANN_LINEAR_FORWARD_FUSED_PATH)
            {
                // todo: implement this
                return cache.a;
            }
            else
            {
                throw std::runtime_error("YANN_LINEAR_FORWARD_FUSED_PATH weren't compiled");
            }
        }
        else
        {
            if constexpr(YANN_LINEAR_FORWARD_REFERENCE_PATH)
            {

                YANN_LOG(3, "cache.a = weights_.values * input", "");
                YANN_LOG(4, "X: {}x{}, | W: {}x{}", input.rows(), input.cols(), weights_.values.rows(), weights_.values.cols());
                cache.a = weights_.values * input;


                YANN_LOG(3, "cache.a = cache.z + biases_.values", "");
                YANN_LOG(4, "A: {}x{} | B: {}x{}", cache.a.rows(), cache.a.cols(), biases_.values.rows(), biases_.values.cols());
                cache.a = cache.a + biases_.values; // with broadcast

                return cache.a;
            }
            else
            {
                throw std::runtime_error("YANN_LINEAR_FORWARD_REFERENCE_PATH weren't compiled");
            }
        }
    }

    cum::Tensor Linear::backward(const cum::Tensor& deltaOutput)
    {
        constexpr bool YANN_LINEAR_BACKWARD_FUSED_PATH = true;
        constexpr bool YANN_LINEAR_BACKWARD_REFERENCE_PATH = true;

        if(runtime_config::fused_kernels())
        {
            if constexpr(YANN_LINEAR_BACKWARD_FUSED_PATH)
            {
                // todo: implement this
            }
            else
            {
                throw std::runtime_error("YANN_LINEAR_BACKWARD_FUSED_PATH weren't compiled");
            }
        }
        else
        {
            if constexpr(YANN_LINEAR_BACKWARD_REFERENCE_PATH)
            {
                YANN_LOG(4, "dL/dZ = deltaOutput * ones", "");
                cache.dz = deltaOutput;

                YANN_LOG(4, "dL/B = cache.dz * cache.x.transpose()", "");
                biases_.gradient += cache.dz.rowwise_sum();

                YANN_LOG(4, "dL/dW = cache.dz * cache.x.transpose()", "");
                weights_.gradient += cache.dz * cache.x.transpose();

                return weights_.values.transpose() * cache.dz;
            }
            else
            {
                throw std::runtime_error("YANN_LINEAR_FORWARD_REFERENCE_PATH weren't compiled");
            }
        }
    }

    void Linear::collect_parameters(std::vector<Parameter*>& params)
    {
        params.push_back(&weights_);
        params.push_back(&biases_);
    }

    std::unique_ptr<LayerBase> Linear::createUnique(int layerSize)
    {
        return std::make_unique<Linear>(layerSize);
    }
} // yann