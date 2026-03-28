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
        // if(inputWidth > 0)
        // {
        //     initParameters(layerSize, inputWidth);
        // }
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

        // std::cout << "input "   << input.rows()   << "x" << input.cols()   << '\n'
        //           << "weights " << weights.rows() << "x" << weights.cols() << '\n'
        //           << "outputs " << outputs.rows() << "x" << outputs.cols() << '\n'
        //           << "=============================================\n";
        // vector_t result = (weights * input) + biases;

        this->inputs = input;
 
        // outputs = (weights * input).colwise() + biases;
        matrix_t weightedSums = (weights * input).colwise() + biases;
        preactivatedOutputs = weightedSums;
        outputs = activation.matrixFunction(weightedSums);


        return outputs;
    }

    matrix_t Dense::backward(const matrix_t& deltaOutput)
    {
    //     std::cout << "d_output " << deltaOutput.rows() << "x" << deltaOutput.cols() << "\n";
    //     std::cout << "outputs  " << outputs.rows() << "x" << outputs.cols() << "\n";


        // Eigen::MatrixXf d_pre_activation = deltaOutput.cwiseProduct(outputs.unaryExpr(activation.derivative));
        matrix_t d_pre_activation = deltaOutput.cwiseProduct(activation.matrixDerivative(preactivatedOutputs));
        // std::cout << "d_pre_activation:\n" << d_pre_activation << '\n';
        // Compute gradients for weights and biases
        // std::cout << "d_pre " << Utils::logs::show_matrix_dimensions(d_pre_activation) << '\n';
        // std::cout << "input " << Utils::logs::show_matrix_dimensions(input) << '\n';
        deltaWeights = d_pre_activation * inputs.transpose();
        // std::cout << "d_weights:\n" << deltaWeights << '\n';
        // std::cout << "2\n";
        deltaBiases = d_pre_activation.rowwise().sum();
        // std::cout << "d_biases:\n" << deltaBiases << '\n';
        // std::cout << "3\n";
        
        
        // Compute gradient w.r.t. input for backpropagation
        matrix_t deltaInput = weights.transpose() * d_pre_activation;       
        // std::cout << "d_input:\n" << deltaBiases << '\n';
        // std::cout << "4\n"; 
        
        return deltaInput;
    }

    void Dense::update_weights(numeric_t rate)
    {
        // std::cout << "weights " << Utils::logs::show_matrix_dimensions(weights) << '\n';
        // std::cout << "weights\n" << weights << '\n';
        // std::cout << "weights grad" << Utils::logs::show_matrix_dimensions(d_weights) << '\n';
        // std::cout << "weights grad\n" << deltaWeights << '\n';
        this->weights -= this->deltaWeights * rate;
        // std::cout << "biases " << Utils::logs::show_matrix_dimensions(biases) << '\n';
        // std::cout << "biases\n" << biases << '\n';
        // std::cout << "biases grad\n" << Utils::logs::show_matrix_dimensions(d_biases) << '\n';
        // std::cout << "biases grad\n" << deltaBiases << '\n';
        this->biases  -= this->deltaBiases  * rate;

        this->deltaWeights = matrix_t::Zero(deltaWeights.rows(), deltaWeights.cols());
        this->deltaBiases = vector_t::Zero(deltaBiases.size());
    }

    std::unique_ptr<LayerBase> Dense::createUnique(int layerSize, const char* func)
    {
        return std::make_unique<Dense>(layerSize, func);
    }
}