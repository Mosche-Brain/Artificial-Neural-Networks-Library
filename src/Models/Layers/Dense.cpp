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
        // activation  = Utils::Activation(func);
        activation  = Utils::getActivationByName(func);
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

        preactivatedOutputs = weights * input;

        outputs = preactivatedOutputs.transform(cum::LinearAlgebra::relu);
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
        
        
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 4)
            std::cout << "\t\t\t\t" << "deltaWeights = matrixMultiply(d_pre_activation, matrixTranspose(inputs))\n";
        #endif
        deltaWeights = math_api::matrixMultiply(d_pre_activation, math_api::matrixTranspose(inputs));

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 4)
            std::cout << "\t\t\t\t" << "deltaBiases = matrixRowwiseSum(d_pre_activation)\n";
        #endif
        deltaBiases = math_api::matrixRowwiseSum(d_pre_activation);
           

        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 4)
            std::cout << "\t\t\t\t" << "deltaInput = matrixMultiply(matrixTranspose(weights), d_pre_activation)\n";
        #endif
        cum::Matrix deltaInput = math_api::matrixMultiply(math_api::matrixTranspose(weights), d_pre_activation);       
        
        return deltaInput;
    }

    void Dense::update_weights(numeric_t rate)
    {
        this->weights -= math_api::matrixScalarMultiply(this->deltaWeights, rate);
        this->biases  -= math_api::vectorScalarMultiply(this->deltaBiases, rate);

        this->deltaWeights = cum::Matrix::Zero(math_api::matrixRows(this->deltaWeights), math_api::matrixCols(this->deltaWeights));
        this->deltaBiases = vector_t::Zero(math_api::vectorSize(this->deltaBiases));
    }

    std::unique_ptr<LayerBase> Dense::createUnique(int layerSize, const char* func)
    {
        return std::make_unique<Dense>(layerSize, func);
    }
}