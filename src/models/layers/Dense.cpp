#include "Dense.hpp"
#include <cum/functions.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/memory.hpp>
#include <cum/neural_primitives/neural_kernels.hpp>

#if defined(ENABLE_DEBUG_OUTPUT)
    #include <iostream>
    #include "utils/formating.hpp"
#endif

#include "runtime_config.hpp"
#include "cum/runtime.hpp"
#include "cum/functions/transform.hpp"
//
// #define ENABLE_RUNTIME_CHECKS // this macro will be moved to runtime config soon
// #define USE_FUSED_KERNELS // this also
// #define ENABLE_CACHED_PREACTIVATION//
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
    
    cum::Matrix Dense::forward(const cum::Matrix& input)
    {
        if(input.rows() != weights.cols())
        {
            if constexpr(ENABLE_RUNTIME_CHECKS)
            {
                // std::cout << "\x1B[31minput size doesn't match with weights\x1B[37m\n";
                    // std::cout << "input " << utils::formating::show_matrix_dimensions(input) << ", "
                        //           << "weights " << utils::formating::show_matrix_dimensions(weights) << '\n';
                            // if(input.cols() != weights.rows())
                if(input.rows() != weights.cols())
                {
                    throw std::runtime_error("Input dimension mismatch: " + std::to_string(input.rows()) + " != " + std::to_string(weights.cols()));
                }
            }
        }

        if (input.cols() != inputs.cols())
        {
            inputs = input;
            raw_outputs = cum::Matrix(outputs.rows(), input.cols());
            outputs(outputs.rows(), outputs.cols());
        }


        cum::runtime::sync();
        if(input.cols() == 1) // single sample
        {
            YANN_LOG(4, "Copying inputs", "");
            // this->inputs = input;
            if constexpr (USE_FUSED_KERNELS)
            {
                if constexpr (ENABLE_CACHED_PREACTIVATION) // must be enabled for proper training in most of cases
                {
                    // cum::functions::various::fill(raw_outputs.data(), 0_c, raw_outputs.size());
                    cum::neural_primitives::neural_kernels::feed_forward_cached_raw(outputs.data(), raw_outputs.data(), weights.values.data(), input.data(), biases.values.data(), weights.values.cols(), weights.values.rows(), activation.name);
                    // cum::neural_primitives::neural_kernels::feed_forward_cached_raw(outputs.data(), raw_outputs.data(), weights().data(), inputs.data(), biases().data(), weights().cols(), weights().rows(), activation.name);
                }
                else // optional for a bit faster inference speed
                {
                    cum::neural_primitives::neural_kernels::feed_forward(outputs.data(), weights().data(), input.data(), biases().data(), weights().cols(), weights().rows(), activation.name);
                }
            }
            else
            {
                YANN_LOG(4, "Performing (weights * input) + biases", "");



                // raw_outputs = (weights() * input) + biases();
                raw_outputs = (weights() * input);
                cum::runtime::sync();
                raw_outputs += biases();
                cum::runtime::sync();

                // cum::Matrix result(raw_outputs.rows(), raw_outputs.cols());
                // cum::runtime::sync();

                YANN_LOG(4, "Performing activation", "");
                // cum::functions::transform(outputs.data(), raw_outputs.data(), raw_outputs.size(), activation);
                cum::functions::transform(outputs.data(), raw_outputs.data(), raw_outputs.size(), activation.name);
            }

            YANN_LOG(4, "forward pass succeed", "");
            return outputs;
        }
        else // batch
        {
            // cum::Matrix results_raw(outputs.rows(), input.cols()); // every column is one batch
            // cum::Matrix results_activated(outputs.rows(), input.cols()); // every column is one batch

            // realocate internal outputs and inputs cache if dimensions mismatch


            // cum::neural_primitives::neural_kernels::feed_forward_cached_raw(results_activated.data(), results_raw.data(), weights.values.data(), input.data(), biases.values.data(), weights.values.cols(), weights.values.rows(), input.cols(), activation.name);
            cum::neural_primitives::neural_kernels::feed_forward_cached_raw(outputs.data(), raw_outputs.data(), weights.values.data(), input.data(), biases.values.data(), weights.values.cols(), weights.values.rows(), input.cols(), activation.name);



            cum::runtime::sync();

            // inputs = input;
            // raw_outputs = results_raw;
            // outputs = results_activated;

            // return results_activated;
            return outputs;
        }

    }

    cum::Matrix Dense::backward(const cum::Matrix& deltaOutput)
    {
        constexpr bool batched = true;
        if constexpr (batched)
        {
            cum::Matrix activation_derivative(raw_outputs.rows(), raw_outputs.cols());
            cum::functions::transform_deriv(activation_derivative.data(), raw_outputs.data(), raw_outputs.size(), activation);

            cum::Matrix cached_somewhat = activation_derivative.cwiseProduct(deltaOutput);

            // for (int i = 0 ; i < inputs.cols() ; i++)
            // {
            //     weights.gradient += cached_somewhat.col(i) * inputs.col(i).transpose();
            //     biases.gradient += cached_somewhat.col(i);
            // }

            weights.gradient = cached_somewhat * inputs.transpose();
            weights.gradient /= inputs.cols();

            biases.gradient = cached_somewhat.rowwiseSum();      
            biases.gradient /= inputs.cols();

            return weights().transpose() * cached_somewhat;
            // biases.gradient = cached_somewhat.rowwiseSum();
        }
        else
        {
            cum::Matrix derivative(raw_outputs.rows(), raw_outputs.cols());
            cum::runtime::sync();

            YANN_LOG(4, "computing activation derivative", "");
            cum::functions::transform_deriv(derivative.data(), raw_outputs.data(), raw_outputs.size(), activation);
            cum::runtime::sync();

            YANN_LOG(4, "td_pre_activation = derivative.cwiseProcut(deltaOutput)\n", "");
            biases.gradient = derivative.cwiseProduct(deltaOutput);
            cum::runtime::sync();

            YANN_LOG(4, "deltaWeights = matrixMultiply(d_pre_activation, matrixTranspose(inputs))\n", "");
            weights.gradient = biases.gradient * inputs.transpose(); // input is col
            cum::runtime::sync();

            return weights().transpose() * biases.gradient;

        }
        // YANN_LOG(4, "deltaBiases = matrixRowwiseSum(d_pre_activation)\n", "");

        // deltaBiases = d_pre_activation.colwiseSum();
        // biases.gradient = d_pre_activation;

        // YANN_LOG(4, "deltaInput = matrixMultiply(matrixTranspose(weights), d_pre_activation)\n", "");

        // cum::Matrix deltaInput = weights().transpose() * biases.gradient;
        // cum::Matrix deltaInput = weights().transpose() * d_pre_activation;
        // cum::Matrix deltaInput = d_pre_activation * weights.transpose();

    }

    void Dense::update_weights(cum::cumeric_t rate) // currently deprecated, now we are using external optimizer - not fixed SGD
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