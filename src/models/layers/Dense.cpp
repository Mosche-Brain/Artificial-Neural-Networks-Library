#include <stdexcept>
#include <print>

#include <cum/memory.hpp>
#include <cum/runtime.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/functions.hpp>
#include <cum/functions/transform.hpp>
#include <cum/neural_primitives/neural_kernels.hpp>
#include <cum/neural_primitives/elementwise.hpp>
#include <cum/neural_primitives/elementwise_diffs.hpp>

#include "yann/runtime_config.hpp"

#include "Dense.hpp"

#define ENABLE_RUNTIME_CHECKS true
#define USE_FUSED_KERNELS false
#define ENABLE_CACHED_PREACTIVATION true

/* TODO:
 * Ograniczenie kopiowania, cache.x może przechywać referencje lub view na cache.a z poprzedniej warstwy
 * Wprowadzenie forward trace selection do runtime config oraz pozostawienie branchy w `if constexpr` by za pomocą konfiguracji cmake można było wyłączyć ich kompilacje
 * Ścieżka z fused kernel dla backward
 * ~~Usunięcie `cum::runtime::sync() pozostałych po debugowaniu~~ (done?)
 * Zwracanie referencji lub widoku do cache przez `forward` i `backward` zamiast kopii
 */

namespace yann::models::layers
{
    Dense::Dense(const int layerSize, const char* func)
    {
        YANN_LOG(1, "Initializing Dense layer with {} neurons and {} activation function...", layerSize, func);
        cum::functions::get_function_by_name(&activation, func);
        _layerSize_ = layerSize;

        this->_layerType_ = LayerType::Dense;
    }

    void Dense::init_parameters(int output_features, int input_features)
    {
        this->weights_       = Parameter::Uniform(output_features, input_features);   /* neurons * input_length */
        this->biases_        = Parameter::Zeros(output_features, 1);                  /* Column-Vector */
        this->cache.a       = cum::Tensor({output_features, 1}, cum::default_type, cum::layout::IO);                 /* Column-Vector */
        this->cache.z   	= cum::Tensor({output_features, 1}, cum::default_type, cum::layout::IO);                 /* Column-Vector */
        this->cache.x       = cum::Tensor({input_features, 1}, cum::default_type, cum::layout::IO);                  /* Column-Vector */
        this->_initialized_ = true;
    }

    
    cum::Tensor Dense::forward(const cum::Tensor& input) // temporary implementation for compatibility
    {
        // those const expresions will be moved to build config
        constexpr bool YANN_DENSE_FORWARD_FUSED_PATH = true;
        constexpr bool YANN_DENSE_FORWARD_REFERENCE_PATH = true;

        this->cache.x = input;

        if(runtime_config::fused_kernels())
        {
            if constexpr(YANN_DENSE_FORWARD_FUSED_PATH)
            {
                // todo: implement this
                return cache.a;
            }
            else
            {
                throw std::runtime_error("YANN_DENSE_FORWARD_FUSED_PATH weren't compiled");
            }
        }
        else
        {
            if constexpr(YANN_DENSE_FORWARD_REFERENCE_PATH)
            {
                cache.z = weights_.values * input;

                cache.z = cache.z + biases_.values; // with broadcast

                cache.a = cache.z.elementwise(activation.name);

                return cache.a;
            }
            else
            {
                throw std::runtime_error("YANN_DENSE_FORWARD_REFERENCE_PATH weren't compiled");
            }
        }
    }

    cum::Tensor Dense::backward(const cum::Tensor& deltaOutput)
    {
        constexpr bool YANN_DENSE_BACKWARD_FUSED_PATH = true;
        constexpr bool YANN_DENSE_BACKWARD_REFERENCE_PATH = true;

        if(runtime_config::fused_kernels())
        {
            if constexpr(YANN_DENSE_BACKWARD_FUSED_PATH)
            {
                // todo: implement this
            }
            else
            {
                throw std::runtime_error("YANN_DENSE_BACKWARD_FUSED_PATH weren't compiled");
            }
        }
        else
        {
            if constexpr(YANN_DENSE_BACKWARD_REFERENCE_PATH)
            {
                YANN_LOG(4, "dA/dZ = cache.z.elementwise_diff(activation.name)", "");
                cache.da = cache.z.elementwise_diff(activation.name);

                YANN_LOG(4, "dL/dZ = deltaOutput * cache.da", "");
                cache.dz = cache.da.cwiseProduct(deltaOutput);

                YANN_LOG(4, "dL/B = cache.dz * cache.x.transpose()", "");
                biases_.gradient += cache.dz.rowwise_sum();

                YANN_LOG(4, "dL/dW = cache.dz * cache.x.transpose()", "");
                weights_.gradient += cache.dz * cache.x.transpose();

                return weights_.values.transpose() * cache.dz;
            }
            else
            {
                throw std::runtime_error("YANN_DENSE_FORWARD_REFERENCE_PATH weren't compiled");
            }
        }
    }


    // cum::Matrix Dense::forward(const cum::Matrix& input)
    // {
    //     if constexpr(ENABLE_RUNTIME_CHECKS)
    //     {
    //         if(input.rows() != weights_.cols())
    //         {
    //             throw std::runtime_error("Input dimension mismatch: " + std::to_string(input.rows()) + " != " + std::to_string(weights_.cols()));
    //         }
    //     }
    //
    //     if (input.cols() != cache.x.cols())
    //     {
    //         cache.resize(cache.x.rows(), cache.z.rows(), input.cols());
    //     }
    //
    //     YANN_LOG(4, "caching inputs", "");
    //     cache.x = input;
    //
    //     cum::runtime::sync();
    //
    //     if constexpr (USE_FUSED_KERNELS)
    //     {
    //         if constexpr (ENABLE_CACHED_PREACTIVATION) // must be enabled for proper training in most of cases
    //         {
    //             YANN_LOG(4, "Running feed forward kernel (cached raw)", "");
    //             cum::neural_primitives::neural_kernels::feed_forward_cached_raw(cache.a.data(), cache.z.data(), weights_.values.data(), input.data(), biases_.values.data(), weights_.values.cols(), weights_.values.rows(), input.cols(), activation.name);
    //         }
    //         else
    //         {
    //             YANN_LOG(4, "Running feed forward kernel", "");
    //             cum::neural_primitives::neural_kernels::feed_forward(cache.a.data(), weights_().data(), input.data(), biases_().data(), weights_().cols(), weights_().rows(), input.cols(), activation.name);
    //         }
    //     }
    //     else
    //     {
    //         YANN_LOG(4, "Performing (weights_ * input) + biases_", "");
    //         cache.z = weights_() * cache.x;
    //         cum::runtime::sync();
    //         cum::LinearAlgebra::addRowVectorInPlace(
    //             cache.z.data(),
    //             biases_().data(),
    //             cache.z.rows(),
    //             cache.z.cols()); // broadcast biasu
    //
    //         cum::runtime::sync();
    //
    //         YANN_LOG(4, "Performing activation", "");
    //         cum::functions::transform(cache.a.data(), cache.z.data(), cache.z.size(), activation.name);
    //     }
    //
    //     YANN_LOG(4, "forward pass succeed", "");
    //     cum::runtime::sync();
    //     return cache.a;
    // }

    // cum::Matrix Dense::backward(const cum::Matrix& deltaOutput)
    // {
    //     cum::runtime::sync();
    //     YANN_LOG(4, "computing activation derivative", "");
    //     cum::functions::transform_deriv(
    //         cache.dz.data(), cache.z.data(), cache.z.size(), activation.name);
    //     cum::runtime::sync();
    //
    //     YANN_LOG(4, "applying chain rule\n", "");
    //     cum::Matrix ΔZ = cache.dz.cwiseProduct(deltaOutput);
    //     cum::runtime::sync();
    //
    //     YANN_LOG(4, "computing weights grad\n", "");
    //     weights_.gradient += ΔZ * cache.x.transpose();
    //     cum::runtime::sync();
    //
    //     YANN_LOG(4, "computing biases grad\n", "");
    //     biases_.gradient += ΔZ.rowwiseSum();
    //     cum::runtime::sync();
    //
    //     return weights_().transpose() * ΔZ;
    //
    // }

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
