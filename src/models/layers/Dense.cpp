
#include "Dense.hpp"
#include <cum/functions.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/memory.hpp>
#include <cum/neural_primitives/neural_kernels.hpp>

#include "runtime_config.hpp"
#include "cum/runtime.hpp"
#include "cum/functions/transform.hpp"

#define ENABLE_RUNTIME_CHECKS true
#define USE_FUSED_KERNELS false
#define ENABLE_CACHED_PREACTIVATION true

namespace yann::models::layers
{
    // Dense::Dense(int layerSize, int inputWidth, const char* func) : LayerBase()
    Dense::Dense(const int layerSize, const char* func)
    {
        YANN_LOG(1, "Initializing Dense layer with {} neurons and {} activation function...", layerSize, func);
        cum::functions::get_function_by_name(&activation, func);
        _layerSize_ = layerSize;

        this->_layerType_ = LAYER_TYPE::DENSE;
    }

    void Dense::initParameters(int output_features, int input_features)
    {
        this->weights_       = Parameter::Uniform(output_features, input_features);   /* neurons * input_length */
        this->biases_        = Parameter::Zeros(output_features, 1);               /* Column-Vector */
        this->cache.a       = cum::Matrix::Zeros(output_features, 1);             /* Column-Vector */
        this->cache.z   	= cum::Matrix::Zeros(output_features, 1);             /* Column-Vector */
        this->cache.x       = cum::Matrix::Zeros(input_features, 1);   /* Column-Vector */
        this->_initialized_ = true;
    }

    cum::Matrix Dense::forward(const cum::Matrix& input) // rozważył bym przekazywanie referencji do wyniku zamiast kopii
    {
        if constexpr(ENABLE_RUNTIME_CHECKS) // może zmienie obecne constexpr na runtime config
        {
            if(input.rows() != weights_.cols())
            {
                throw std::runtime_error("Input dimension mismatch: " + std::to_string(input.rows()) + " != " + std::to_string(weights_.cols()));
            }
        }

        if (input.cols() != cache.x.cols()) // sprawdza czy batch jest taki sam
        {
            cache.resize(cache.x.rows(), cache.z.rows(), input.cols());
        }

        YANN_LOG(4, "caching inputs", "");
        cache.x = input;

        cum::runtime::sync();

        if constexpr (USE_FUSED_KERNELS)
        {
            if constexpr (ENABLE_CACHED_PREACTIVATION) // must be enabled for proper training in most of cases
            {
                YANN_LOG(4, "Running feed forward kernel (cached raw)", "");
                // cum::neural_primitives::neural_kernels::feed_forward_cached_raw(cache.a.data(), cache.z.data(), weights_.values.data(), input.data(), biases_.values.data(), weights_.values.cols(), weights_.values.rows(), activation.name);
                cum::neural_primitives::neural_kernels::feed_forward_cached_raw(cache.a.data(), cache.z.data(), weights_.values.data(), input.data(), biases_.values.data(), weights_.values.cols(), weights_.values.rows(), input.cols(), activation.name);

            }
            else // for a bit faster inference speed (training may not be posible in some cases)
            {
                YANN_LOG(4, "Running feed forward kernel", "");
                cum::neural_primitives::neural_kernels::feed_forward(cache.a.data(), weights_().data(), input.data(), biases_().data(), weights_().cols(), weights_().rows(), input.cols(), activation.name);
            }
        }
        else
        {
            YANN_LOG(4, "Performing (weights_ * input) + biases_", "");
            // cache.z = (weights_() * input) += biases_();;
            cache.z = weights_() * cache.x;
            cum::runtime::sync();
            cum::LinearAlgebra::addRowVectorInPlace(
                cache.z.data(),
                biases_().data(),
                cache.z.rows(),
                cache.z.cols());

            cum::runtime::sync();

            YANN_LOG(4, "Performing activation", "");
            cum::functions::transform(cache.a.data(), cache.z.data(), cache.z.size(), activation.name);
        }

        YANN_LOG(4, "forward pass succeed", "");
        cum::runtime::sync();
        return cache.a;
    }

    cum::Matrix Dense::backward(const cum::Matrix& deltaOutput)
    {
        constexpr bool batched = true;
        if constexpr (batched)
        {
            cum::runtime::sync();
            cum::functions::transform_deriv(
                cache.dz.data(), cache.z.data(), cache.z.size(), activation.name);
            cum::runtime::sync();

            cum::Matrix cached_somewhat = cache.dz.cwiseProduct(deltaOutput);
            cum::runtime::sync();

            weights_.gradient += cached_somewhat * cache.x.transpose();
            cum::runtime::sync();

            biases_.gradient += cached_somewhat.rowwiseSum();
            cum::runtime::sync();


            return weights_().transpose() * cached_somewhat;
        }
        else
        {
            cum::Matrix derivative(cache.z.rows(), cache.z.cols());
            cum::runtime::sync();

            YANN_LOG(4, "computing activation derivative", "");
            cum::functions::transform_deriv(derivative.data(), cache.z.data(), cache.z.size(), activation);
            cum::runtime::sync();

            YANN_LOG(4, "td_pre_activation = derivative.cwiseProcut(deltaOutput)\n", "");
            biases_.gradient = derivative.cwiseProduct(deltaOutput);
            cum::runtime::sync();

            YANN_LOG(4, "deltaWeights = matrixMultiply(d_pre_activation, matrixTranspose(inputs))\n", "");
            weights_.gradient = biases_.gradient * cache.x.transpose();
            cum::runtime::sync();

            return weights_().transpose() * biases_.gradient;
        }
    }

    void Dense::collect_parameters(std::vector<Parameter*>& params)
    {
        params.push_back(&weights_);
        params.push_back(&biases_);
    }



    std::unique_ptr<LayerBase> Dense::createUnique(int layerSize, const char* func)
    {
        return std::make_unique<Dense>(layerSize, func);
    }
}
