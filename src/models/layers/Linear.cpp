//
// Created by jaro on 8/31/26.
//

#include "yann/models/layers/Linear.hpp"

#include "cum/runtime.hpp"
#include "cum/functions/transform.hpp"
#include "yann/logging/Logger.hpp"

#define ENABLE_RUNTIME_CHECKS true

namespace yann::models::layers
{
    Linear::Linear(const int layerSize)
    {
        YANN_LOG(1, "Initializing Linear layer with {} neurons...", layerSize);
        _layerSize_ = layerSize;

        this->_layerType_ = LAYER_TYPE::LINEAR;
    }

    void Linear::init_parameters(int output_features, int input_features)
    {
        this->weights_      = Parameter::Uniform(output_features, input_features);   /* neurons * input_length */
        this->biases_       = Parameter::Zeros(output_features, 1);               /* Column-Vector */
        this->cache.x       = cum::Tensor::Zeros({input_features, 1});   /* Column-Vector */
        this->cache.z   	= cum::Tensor::Zeros({output_features, 1});             /* Column-Vector */
        this->cache.dz      = cum::Tensor::Ones({output_features, 1});
        this->_initialized_ = true;
    }

    cum::Tensor Linear::forward(const cum::Tensor& input)
    {
        // if constexpr(ENABLE_RUNTIME_CHECKS)
        // {
        //     if(input.rows() != weights_.cols())
        //     {
        //         throw std::runtime_error("Input dimension mismatch: " + std::to_string(input.rows()) + " != " + std::to_string(weights_.cols()));
        //     }
        // }
        //
        // if (input.cols() != cache.x.cols())
        // {
        //     cache.resize(cache.x.rows(), cache.z.rows(), input.cols());
        //     cache.dz = cum::Tensor::Zeros({cache.z.rows(), input.cols()}, cum::default_type, cum::layout::NC);
        // }
        //
        // YANN_LOG(4, "caching inputs", "");
        // cache.x = input;

        cum::runtime::sync();
        //
        // if constexpr (USE_FUSED_KERNELS)
        // {
        //     if constexpr (ENABLE_CACHED_PREACTIVATION) // must be enabled for proper training in most of cases
        //     {
        //         YANN_LOG(4, "Running feed forward kernel (cached raw)", "");
        //         // cum::neural_primitives::neural_kernels::feed_forward_cached_raw(cache.a.data(), cache.z.data(), weights_.values.data(), input.data(), biases_.values.data(), weights_.values.cols(), weights_.values.rows(), activation.name);
        //         cum::neural_primitives::neural_kernels::feed_forward_cached_raw(cache.a.data(), cache.z.data(), weights_.values.data(), input.data(), biases_.values.data(), weights_.values.cols(), weights_.values.rows(), input.cols(), activation.name);
        //
        //     }
        //     else // for a bit faster inference speed (training may not be posible in some cases)
        //     {
        //         YANN_LOG(4, "Running feed forward kernel", "");
        //         cum::neural_primitives::neural_kernels::feed_forward(cache.a.data(), weights_().data(), input.data(), biases_().data(), weights_().cols(), weights_().rows(), input.cols(), activation.name);
        //     }
        // }
        // else
        // {
        //     YANN_LOG(4, "Performing (weights_ * input) + biases_", "");
        //     // cache.z = (weights_() * input) += biases_();;
        //     cache.z = weights_() * cache.x;
        //     cum::runtime::sync();
        //     cum::LinearAlgebra::addRowVectorInPlace(
        //         cache.z.data<cum::cumeric_t>(),
        //         biases_().data<cum::cumeric_t>(),
        //         cache.z.rows(),
        //         cache.z.cols());
        //
        //     cum::runtime::sync();
        //
        // // }
        //
        // YANN_LOG(4, "forward pass succeed", "");
        // cum::runtime::sync();
        // return cache.z;
    }

    cum::Tensor Linear::backward(const cum::Tensor& deltaOutput)
    {
        // cum::runtime::sync();
        // cum::functions::transform_deriv(
        //     cache.dz.data(), cache.z.data(), cache.z.size(), cum::functions::function_id::linear);
        // cum::runtime::sync();

        // cum::Matrix cached_somewhat = cache.dz.cwiseProduct(deltaOutput);
        // cum::runtime::sync();
        //
        // weights_.gradient += cached_somewhat * cache.x.transpose();
        // cum::runtime::sync();
        //
        // biases_.gradient += cached_somewhat.rowwiseSum();
        // cum::runtime::sync();
        //
        //
        // return weights_().transpose() * cached_somewhat;
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