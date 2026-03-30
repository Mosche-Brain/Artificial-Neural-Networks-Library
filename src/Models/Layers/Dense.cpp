#include "Dense.hpp"

#include <iostream>

#include "Utility/logs.hpp"

namespace YANN::Models::Layers
{
    // Dense::Dense(int layerSize, int inputWidth, const char* func) : LayerBase()
    Dense::Dense(int layerSize, const char* func)
    {
        activation  = Utils::Activation(func);
        _layerSize_ = layerSize;

        this->_layerType_   = LAYER_TYPE::DENSE;
    }
    
    matrix_t Dense::forward(const matrix_t& input)
    {
        if(input.size() != weights.cols())
        {
            // std::cout << "\x1B[31minput size doesn't match with weights\x1B[37m\n";
            // std::cout << "input " << Utils::logs::show_matrix_dimensions(input) << ", "
                    //   << "weights " << Utils::logs::show_matrix_dimensions(weights) << '\n';
        }

        this->inputs = input;
 
        preactivatedOutputs = math_api::matrixColwiseAdd(math_api::matrixMultiply(weights, input), biases);
        // outputs = math_api::matrixTransform(preactivatedOutputs, activation.function);
        outputs = activation.matrixFunction(preactivatedOutputs);

        return outputs;
    }

    matrix_t Dense::backward(const matrix_t& deltaOutput)
    {
        matrix_t d_pre_activation = math_api::matrixElementwiseMultiply(deltaOutput, activation.matrixDerivative(preactivatedOutputs));
        deltaWeights = math_api::matrixMultiply(d_pre_activation, math_api::matrixTranspose(inputs));
        deltaBiases = math_api::matrixRowwiseSum(d_pre_activation);
           
        matrix_t deltaInput = math_api::matrixMultiply(math_api::matrixTranspose(weights), d_pre_activation);       
        
        return deltaInput;
    }

    void Dense::update_weights(numeric_t rate)
    {
        this->weights -= math_api::matrixScalarMultiply(this->deltaWeights, rate);
        this->biases  -= math_api::vectorScalarMultiply(this->deltaBiases, rate);

        this->deltaWeights = matrix_t::Zero(math_api::matrixRows(this->deltaWeights), math_api::matrixCols(this->deltaWeights));
        this->deltaBiases = vector_t::Zero(math_api::vectorSize(this->deltaBiases));
    }

    std::unique_ptr<LayerBase> Dense::createUnique(int layerSize, const char* func)
    {
        return std::make_unique<Dense>(layerSize, func);
    }
}