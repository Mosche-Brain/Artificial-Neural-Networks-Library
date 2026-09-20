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
 * Brak niejawnej alokacji cache potrzebnego tylko do treningu
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
        this->weights_      = Parameter::Uniform(output_features, input_features);   /* neurons * input_length */
        this->biases_       = Parameter::Zeros(output_features, 1);                  /* Column-Vector */
        this->cache.a       = cum::Tensor({output_features, 1}, cum::default_type, cum::layout::IO);                 /* Column-Vector */
        this->cache.z   	= cum::Tensor({output_features, 1}, cum::default_type, cum::layout::IO);                 /* Column-Vector */
        this->cache.x       = cum::Tensor({input_features, 1}, cum::default_type, cum::layout::IO);                  /* Column-Vector */
        this->cache.da      = cum::Tensor({output_features, 1}, cum::default_type, cum::layout::IO);                 /* Column-Vector */
        this->_initialized_ = true;
    }

    
    cum::Tensor Dense::forward(const cum::Tensor& input) // temporary implementation for compatibility
    {
        // those const expresions will be moved to build config
        constexpr bool YANN_DENSE_FORWARD_FUSED_PATH = true;
        constexpr bool YANN_DENSE_FORWARD_REFERENCE_PATH = true;

        // check if number of weights cols is equal to number of input rows
        if constexpr(ENABLE_RUNTIME_CHECKS)
        {
            if(input.rows() != weights_.cols())
            {
                throw std::runtime_error("Input dimension mismatch: " + std::to_string(input.rows()) + " != " + std::to_string(weights_.cols()));
            }
        }

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
                YANN_LOG(3, "cache.z = weights_.values * input", "");
                YANN_LOG(4, "X: {}x{}, | W: {}x{}", input.rows(), input.cols(), weights_.values.rows(), weights_.values.cols());
                cache.z = weights_.values * input;

                YANN_LOG(3, "cache.z = cache.z + biases_.values", "");
                YANN_LOG(4, "Z: {}x{} | B: {}x{}", cache.z.rows(), cache.z.cols(), biases_.values.rows(), biases_.values.cols());
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
            if constexpr(YANN_DENSE_BACKWARD_REFERENCE_PATH) // We may move transpositions to compute kernel, but for now it's fine
            {
                YANN_LOG(3, "dA/dZ = cache.z.elementwise_diff(activation.name)", "");
                cache.da = cache.z.elementwise_diff(activation.name);

                YANN_LOG(3, "dL/dZ = deltaOutput * cache.da", "");
                cache.dz = cache.da.cwiseProduct(deltaOutput);

                YANN_LOG(3, "dL/dB = rowwise sum of cache.dz", "");
                // if (cache.dz.cols() == 1)
                    // biases_.gradient += cache.dz;
                // else
                    biases_.gradient += cache.dz.rowwise_sum();

                YANN_LOG(3, "dL/dW = cache.dz * cache.x.transpose()", "");
                YANN_LOG(4, "dZ: {}x{} | X: {}x{}", cache.dz.rows(), cache.dz.cols(), cache.x.rows(), cache.x.cols());

                weights_.gradient += cache.dz * cache.x.transpose();

                YANN_LOG(3, "dL/dX = weights.transpose() * cahce.dz", "");
                YANN_LOG(4, "W: {}x{}, | dZ: {}x{}", weights_.values.rows(), weights_.values.cols(), cache.dz.rows(), cache.dz.cols());
                // cum::Tensor transposed_weights = weights_.values.transpose();
                // YANN_LOG(4, "W: {}x{}", transposed_weights.rows(), transposed_weights.cols());

                // return transposed_weights * cache.dz;
                return weights_.values.transpose() * cache.dz;
            }
            else
            {
                throw std::runtime_error("YANN_DENSE_FORWARD_REFERENCE_PATH weren't compiled");
            }
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
