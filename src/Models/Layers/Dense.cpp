#include "Dense.hpp"
#include <cum/functions.hpp>
#include <cum/LinearAlgebra.hpp>

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
        cum::functions::getFunctionByName(&activation, func);
        _layerSize_ = layerSize;

        this->_layerType_   = LAYER_TYPE::DENSE;
    }
    
    cum::Matrix Dense::forward(const cum::Matrix& input)
    {
        if(input.size() != weights.cols())
        {
            // std::cout << "\x1B[31minput size doesn't match with weights\x1B[37m\n";
            // std::cout << "input " << Utils::logs::show_matrix_dimensions(input) << ", "
                    //   << "weights " << Utils::logs::show_matrix_dimensions(weights) << '\n';
        }

        this->inputs = input;
 
        // preactivatedOutputs = math_api::matrixColwiseAdd(math_api::matrixMultiply(weights, input), biases);
        // outputs = math_api::matrixTransform(preactivatedOutputs, activation.function);

        preactivatedOutputs = (weights * input) + biases;

        // outputs = preactivatedOutputs.transform(cum::LinearAlgebra::relu);
        cum::functions::transform(outputs.data(), preactivatedOutputs.data(), activation, preactivatedOutputs.size());
        // preactivatedOutputs = weights

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