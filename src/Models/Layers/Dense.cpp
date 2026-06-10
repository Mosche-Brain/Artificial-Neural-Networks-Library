#include "Dense.hpp"
#include <cum/functions.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/memory.hpp>

#if defined(ENABLE_DEBUG_OUTPUT)
    #include <iostream>
    #include "Utility/logs.hpp"
#endif

#include "runtime_config.hpp"

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
    
    cum::Matrix& Dense::forward(const cum::Matrix& input)
    {
        if(input.size() != weights.cols())
        {
            #if defined(ENABLE_DEBUG_OUTPUT)     
                std::cout << "\x1B[31minput size doesn't match with weights\x1B[37m\n";
                std::cout << "input " << Utils::logs::show_matrix_dimensions(input) << ", "
                          << "weights " << Utils::logs::show_matrix_dimensions(weights) << '\n';
            #endif

        }

        #if defined(ENABLE_DEBUG_OUTPUT) 
            std::cout << "copying inputs\n";
        #endif
        this->inputs = input;

        #if defined(ENABLE_DEBUG_OUTPUT) 
            std::cout << "Performing (weights * input) + biases\n";
        #endif
        // Note: when using row_major + f16 + GPU, the underlying Matrix leading dims must be
        // compatible with what matMul passes to row_major::gemm (lda = logical cols of A).
        // Odd k (like layer width 3) can trigger lda validation errors in hgemm on Arc.
        preactivatedOutputs = (weights * input) + biases;
        
        #if defined(ENABLE_DEBUG_OUTPUT)    
            std::cout << "Performing activation\n";
        #endif
        // cum::functions::transform(outputs.data(), preactivatedOutputs.data(), activation, preactivatedOutputs.size());
        cum::Matrix result(_layerSize_, 1);
        cum::functions::transform(result.data(), preactivatedOutputs.data(), activation, preactivatedOutputs.size());
        outputs = result;
        #if defined(ENABLE_DEBUG_OUTPUT)    
            std::cout << "forward pass succed\n";
        #endif
        return outputs;
    }

    cum::Matrix Dense::backward(const cum::Matrix& deltaOutput)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 4)
            std::cout << "\t\t\t\t" << "matrixElementwiseMultiply(deltaOutput, activation.matrixDerivative(preactivatedOutputs))\n";
        #endif

        // cum::Matrix d_pre_activation = math_api::matrixElementwiseMultiply(deltaOutput, 
        //                                                                 math_api::matrixTransform(preactivatedOutputs, activation.derivative));
        
        // cum::Matrix d_pre_activation = deltaOutput.cwiseProduct(cum::functions::transform(preactivatedOutputs, activation.derivative));
        // cum::Matrix d_pre_activation(deltaOutput.rows(), deltaOutput.cols());
        cum::Matrix d_pre_activation = deltaOutput;
        cum::functions::activationDerivative(d_pre_activation.data(), preactivatedOutputs.data(), activation, preactivatedOutputs.size());
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 4)
            std::cout << "\t\t\t\t" << "deltaWeights = matrixMultiply(d_pre_activation, matrixTranspose(inputs))\n";
        #endif
        // deltaWeights = math_api::matrixMultiply(d_pre_activation, math_api::matrixTranspose(inputs));
        deltaWeights = d_pre_activation * inputs.transpose();

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 4)
            std::cout << "\t\t\t\t" << "deltaBiases = matrixRowwiseSum(d_pre_activation)\n";
        #endif
        // deltaBiases = math_api::matrixRowwiseSum(d_pre_activation);
        deltaBiases = d_pre_activation.rowwiseSum();
           

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 4)
            std::cout << "\t\t\t\t" << "deltaInput = matrixMultiply(matrixTranspose(weights), d_pre_activation)\n";
        #endif
        // cum::Matrix deltaInput = math_api::matrixMultiply(math_api::matrixTranspose(weights), d_pre_activation);       
        cum::Matrix deltaInput = weights.transpose() * d_pre_activation;       
        
        return deltaInput;
    }

    void Dense::update_weights(cum::cumeric_t rate)
    {
        this->weights -= this->deltaWeights * rate;
        this->biases  -= this->deltaBiases  * rate;

        this->deltaWeights = cum::Matrix(this->deltaWeights.rows(), this->deltaWeights.cols(), 0_c);
        this->deltaBiases = cum::Matrix(this->deltaBiases.rows(), this->deltaBiases.cols(), 0_c);
    }

    std::unique_ptr<LayerBase> Dense::createUnique(int layerSize, const char* func)
    {
        return std::make_unique<Dense>(layerSize, func);
    }
}