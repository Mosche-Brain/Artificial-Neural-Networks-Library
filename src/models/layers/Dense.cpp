#include "Dense.hpp"
#include <cum/functions.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/memory.hpp>
#include <cum/neural_primitives/neural_kernels.hpp>

#include "runtime_config.hpp"
#include "cum/runtime.hpp"
#include "cum/functions/transform.hpp"

#define ENABLE_RUNTIME_CHECKS true
#define USE_FUSED_KERNELS true
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

    cum::Matrix Dense::forward(const cum::Matrix& input) // rozważył bym przekazywanie referencji do wyniku zamiast kopii
    {
        if constexpr(ENABLE_RUNTIME_CHECKS) // może zmienie obecne constexpr na runtime config
        {
            if(input.rows() != weights.cols())
            {
                throw std::runtime_error("Input dimension mismatch: " + std::to_string(input.rows()) + " != " + std::to_string(weights.cols()));
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
                // cum::neural_primitives::neural_kernels::feed_forward_cached_raw(cache.a.data(), cache.z.data(), weights.values.data(), input.data(), biases.values.data(), weights.values.cols(), weights.values.rows(), activation.name);
                cum::neural_primitives::neural_kernels::feed_forward_cached_raw(cache.a.data(), cache.z.data(), weights.values.data(), input.data(), biases.values.data(), weights.values.cols(), weights.values.rows(), input.cols(), activation.name);

            }
            else // for a bit faster inference speed (training may not be posible in some cases)
            {
                YANN_LOG(4, "Running feed forward kernel", "");
                cum::neural_primitives::neural_kernels::feed_forward(cache.a.data(), weights().data(), input.data(), biases().data(), weights().cols(), weights().rows(), input.cols(), activation.name);
            }
        }
        else
        {
            YANN_LOG(4, "Performing (weights * input) + biases", "");
            cache.z = (weights() * input) += biases();;

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
            cum::Matrix activation_derivative(cache.z.rows(), cache.z.cols()); // tą macierz mógłbym przenieść do struktury cache by uniknąć alokacji w trakcie wykonywania propagacji wstecznej
            cum::functions::transform_deriv(activation_derivative.data(), cache.z.data(), cache.z.size(), activation);

            cum::Matrix cached_somewhat = activation_derivative.cwiseProduct(deltaOutput); // tą też

            // for (int i = 0 ; i < inputs.cols() ; i++)
            // {
            //     weights.gradient += cached_somewhat.col(i) * inputs.col(i).transpose();
            //     biases.gradient += cached_somewhat.col(i);
            // }

            // weights.gradient = cached_somewhat * cache.x.transpose();
            weights.gradient += cached_somewhat * cache.x.transpose();
            weights.gradient /= cache.x.cols();

            // biases.gradient = cached_somewhat.rowwiseSum();
            biases.gradient += cached_somewhat.rowwiseSum();
            biases.gradient /= cache.x.cols();

            // dodaje gradienty zamiast przypisywać by optymalizator miał dowolność co to tego czy chce je wyzerować czy przeskalować po kroku dostrajania

            return weights().transpose() * cached_somewhat;
        }
        else
        {
            cum::Matrix derivative(cache.z.rows(), cache.z.cols());
            cum::runtime::sync();

            YANN_LOG(4, "computing activation derivative", "");
            cum::functions::transform_deriv(derivative.data(), cache.z.data(), cache.z.size(), activation);
            cum::runtime::sync();

            YANN_LOG(4, "td_pre_activation = derivative.cwiseProcut(deltaOutput)\n", "");
            biases.gradient = derivative.cwiseProduct(deltaOutput);
            cum::runtime::sync();

            YANN_LOG(4, "deltaWeights = matrixMultiply(d_pre_activation, matrixTranspose(inputs))\n", "");
            weights.gradient = biases.gradient * cache.x.transpose(); // input is col
            cum::runtime::sync();

            return weights().transpose() * biases.gradient;

        }
    }

    void Dense::update_weights(cum::cumeric_t rate) // currently deprecated, now we are using external optimizer - not fixed SGD - raczej usunę tą funcje
    {
    //     #if defined(ENABLE_DEBUG_OUTPUT)
    //         cum::Matrix oldWeights = this->weights;
    //     #endif
    //
    //     this->weights -= this->deltaWeights * rate;
    //     this->biases  -= this->deltaBiases  * rate;
    //
    //     #if defined(ENABLE_DEBUG_OUTPUT)
    //     if(runtime_config::verbosity_level() >= 4)
    //         std::cout << utils::formating::matricesWithArrowToString(oldWeights, weights, 4, 16) << '\n';
    //     #endif
    //
    //     this->deltaWeights = cum::Matrix(this->deltaWeights.rows(), this->deltaWeights.cols(), 0_c);
    //     this->deltaBiases = cum::Matrix(this->deltaBiases.rows(), this->deltaBiases.cols(), 0_c);
    }

    void Dense::collect_parameters(std::vector<Parameter*>& params)
    {
        params.push_back(&weights);
        params.push_back(&biases);
    }

    std::unique_ptr<LayerBase> Dense::createUnique(int layerSize, const char* func)
    {
        return std::make_unique<Dense>(layerSize, func);
    }
}