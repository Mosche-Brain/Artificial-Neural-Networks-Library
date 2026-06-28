#include "Dense.hpp"
#include <cum/functions.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/memory.hpp>

#if defined(ENABLE_DEBUG_OUTPUT)
    #include <iostream>
    #include "Utility/logs.hpp"
#endif

#include "runtime_config.hpp"

#define ENABLE_RUNTIME_CHECKS // this macro will be moved to runtime config soon

namespace YANN::Models::Layers
{
    // Dense::Dense(int layerSize, int inputWidth, const char* func) : LayerBase()
    Dense::Dense(int layerSize, const char* func)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        // if(runtime_config::DEBUG_VEBOSITY >= 3)
            std::cout << "\t\t\t" << "Initializing Dense layer with " << layerSize << " neurons and " << func << " activation function...\n";
        #endif
        cum::functions::getFunctionByName(&activation, func);
        _layerSize_ = layerSize;

        this->_layerType_   = LAYER_TYPE::DENSE;
    }
    
    cum::Matrix Dense::forward(const cum::Matrix& input)
    {
        if(input.size() != weights.cols())
        {
            #if defined(ENABLE_RUNTIME_CHECKS)
                // std::cout << "\x1B[31minput size doesn't match with weights\x1B[37m\n";
                // std::cout << "input " << Utils::logs::show_matrix_dimensions(input) << ", "
                //           << "weights " << Utils::logs::show_matrix_dimensions(weights) << '\n';

            throw std::runtime_error("Input dimension mismatch: " +
                std::to_string(input.cols()) + " != " + std::to_string(weights.rows()));
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

        preactivatedOutputs = (input * weights) + biases;

        #if defined(ENABLE_DEBUG_OUTPUT)    
        if(runtime_config::verbosity_level() >= 4)
            std::cout << "Performing activation\n";
        #endif
        // cum::functions::transform(outputs.data(), preactivatedOutputs.data(), activation, preactivatedOutputs.size());
        // cum::Matrix result(_layerSize_, 1);
        cum::Matrix result(preactivatedOutputs.rows(), preactivatedOutputs.cols());
        cum::functions::transform(result.data(), preactivatedOutputs.data(), activation, preactivatedOutputs.size());
        outputs = result;
        #if defined(ENABLE_DEBUG_OUTPUT)   
        if(runtime_config::verbosity_level() >= 4) 
            std::cout << "forward pass succed\n";
        #endif
        return outputs;
    }

    cum::Matrix Dense::backward(const cum::Matrix& deltaOutput)
    {

        // cum::Matrix d_pre_activation = math_api::matrixElementwiseMultiply(deltaOutput, 
        //                                                                 math_api::matrixTransform(preactivatedOutputs, activation.derivative));
        
        // cum::Matrix d_pre_activation = deltaOutput.cwiseProduct(cum::functions::transform(preactivatedOutputs, activation.derivative));
        // cum::Matrix d_pre_activation(deltaOutput.rows(), deltaOutput.cols());
        // cum::Matrix derivative(deltaOutput.cols(), deltaOutput.cols());
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

        cum::Matrix d_pre_activation = derivative.cwiseProduct(deltaOutput);

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 4)
            std::cout << "\t\t\t\t" << "deltaWeights = matrixMultiply(d_pre_activation, matrixTranspose(inputs))\n";
        #endif
        // deltaWeights = d_pre_activation * inputs.transpose(); // input is col
        deltaWeights = inputs.transpose() * d_pre_activation; // input is row

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 4)
            std::cout << "\t\t\t\t" << "deltaBiases = matrixRowwiseSum(d_pre_activation)\n";
        #endif

        // deltaBiases = d_pre_activation.colwiseSum();
        deltaBiases = d_pre_activation;


        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 4)
            std::cout << "\t\t\t\t" << "deltaInput = matrixMultiply(matrixTranspose(weights), d_pre_activation)\n";
        #endif

        // cum::Matrix deltaInput = weights.transpose() * d_pre_activation;
        cum::Matrix deltaInput = d_pre_activation * weights.transpose();

        return deltaInput;
    }

    void Dense::update_weights(cum::cumeric_t rate)
    {
        // const cum::cummulative_t grad_scale = 1024.f; // temporarely hardcoded
        // this->deltaBiases *= (1.f / grad_scale);
        // this->deltaWeights *= (1.f / grad_scale);

        this->weights -= this->deltaWeights * rate;
        this->biases  -= this->deltaBiases  * rate;

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 4)
            // std::cout << "\t\t\t\t" << "changed weight by " << Utils::logs::matrixToString(weights.transpose()) << " * rate\n";
                std::cout << Utils::logs::matricesWithArrowToString(deltaWeights, weights, 4, 16) << '\n';
        #endif

        this->deltaWeights = cum::Matrix(this->deltaWeights.rows(), this->deltaWeights.cols(), 0_c);
        this->deltaBiases = cum::Matrix(this->deltaBiases.rows(), this->deltaBiases.cols(), 0_c);
    }

    std::unique_ptr<LayerBase> Dense::createUnique(int layerSize, const char* func)
    {
        return std::make_unique<Dense>(layerSize, func);
    }
}