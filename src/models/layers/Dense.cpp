#include "Dense.hpp"
#include <cum/functions.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/memory.hpp>
#include <cum/neural_primitives/neural_kernels.hpp>

#if defined(ENABLE_DEBUG_OUTPUT)
    #include <iostream>
    #include "utils/logs.hpp"
#endif

#include "runtime_config.hpp"
#include "cum/functions/transform.hpp"

#define ENABLE_RUNTIME_CHECKS // this macro will be moved to runtime config soon
// #define USE_FUSED_KERNELS // this also
#define ENABLE_CACHED_PREACTIVATION

namespace yann::models::layers
{
    // Dense::Dense(int layerSize, int inputWidth, const char* func) : LayerBase()
    Dense::Dense(int layerSize, const char* func)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 1)
            std::cout << "\t" << "Initializing Dense layer with " << layerSize << " neurons and " << func << " activation function...\n";
        #endif
        cum::functions::getFunctionByName(&activation, func);
        _layerSize_ = layerSize;

        this->_layerType_ = LAYER_TYPE::DENSE;
    }
    
    cum::Matrix Dense::forward(const cum::Matrix& input)
    {
        if(input.size() != weights.cols())
        {
            #if defined(ENABLE_RUNTIME_CHECKS)
                // std::cout << "\x1B[31minput size doesn't match with weights\x1B[37m\n";
                // std::cout << "input " << utils::logs::show_matrix_dimensions(input) << ", "
                //           << "weights " << utils::logs::show_matrix_dimensions(weights) << '\n';
            // if(input.cols() != weights.rows())
            if(input.rows() != weights.cols())
            {
                throw std::runtime_error("Input dimension mismatch: " + std::to_string(input.rows()) + " != " + std::to_string(weights.cols()));
            }
            #endif
        }

        #if defined(ENABLE_DEBUG_OUTPUT) 
        if(runtime_config::verbosity_level() >= 4)
            std::cout << "copying inputs\n";
        #endif
        this->inputs = input;

        #if defined(ENABLE_DEBUG_OUTPUT) 
        if(runtime_config::verbosity_level() >= 4)
            std::cout << "Performing (weights * input) + biases\n";
        #endif

        #if defined(USE_FUSED_KERNELS)
            #if defined(ENABLE_CACHED_PREACTIVATION)
            cum::neural_primitives::neural_kernels::feed_forward(preactivatedOutputs.data(), weights().data(), input.data(), biases().data(), weights().cols(), weights().rows());
            cum::functions::transform(result.data(), preactivatedOutputs.data(), preactivatedOutputs.size(), activation);
            #else
            cum::neural_primitives::neural_kernels::feed_forward(outputs.data(), weights().data(), input.data(), biases().data(), weights().cols(), weights().rows(), activation.name);
            #endif
        #else
        preactivatedOutputs = (weights() * input) + biases();
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 4)
            std::cout << "Performing activation\n";
        #endif

        cum::Matrix result(preactivatedOutputs.rows(), preactivatedOutputs.cols());
        cum::functions::transform(result.data(), preactivatedOutputs.data(), preactivatedOutputs.size(), activation);
        outputs = result;
        #endif

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 4) 
            std::cout << "forward pass succed\n";
        #endif
        return outputs;
    }

    cum::Matrix Dense::backward(const cum::Matrix& deltaOutput)
    {
        cum::Matrix derivative(preactivatedOutputs.rows(), preactivatedOutputs.cols());

        #if defined(ENABLE_DEBUG_OUTPUT)
            if(runtime_config::verbosity_level() >= 4)
                std::cout << "\t\t\t\t" << "computing activation derviative\n";
        #endif

        cum::functions::activationDerivative(derivative.data(), preactivatedOutputs.data(), activation, preactivatedOutputs.size());

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 4)
            std::cout << "\t\t\t\t" << "d_pre_activation = derivative.cwiseProcut(deltaOutput)\n";
        #endif

        // cum::Matrix d_pre_activation = derivative.cwiseProduct(deltaOutput);
        biases.gradient = derivative.cwiseProduct(deltaOutput);

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 4)
            std::cout << "\t\t\t\t" << "deltaWeights = matrixMultiply(d_pre_activation, matrixTranspose(inputs))\n";
        #endif
        weights.gradient = biases.gradient * inputs.transpose(); // input is col
        // weights.gradient = d_pre_activation * inputs.transpose(); // input is col
        // deltaWeights = inputs.transpose() * d_pre_activation; // input is row

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 4)
            std::cout << "\t\t\t\t" << "deltaBiases = matrixRowwiseSum(d_pre_activation)\n";
        #endif

        // deltaBiases = d_pre_activation.colwiseSum();
        // biases.gradient = d_pre_activation;

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 4)
            std::cout << "\t\t\t\t" << "deltaInput = matrixMultiply(matrixTranspose(weights), d_pre_activation)\n";
        #endif

        // cum::Matrix deltaInput = weights().transpose() * biases.gradient;
        // cum::Matrix deltaInput = weights().transpose() * d_pre_activation;
        // cum::Matrix deltaInput = d_pre_activation * weights.transpose();

        return weights().transpose() * biases.gradient;
    }

    void Dense::update_weights(cum::cumeric_t rate) // currently depraced, now we are using external optimizer - not fixed SGD
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
    //         std::cout << utils::logs::matricesWithArrowToString(oldWeights, weights, 4, 16) << '\n';
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