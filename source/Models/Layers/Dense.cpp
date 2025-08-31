#include "Dense.hpp"

#include <iostream>

namespace SNN::Models::Layers
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
    }
    
    Eigen::MatrixXf Dense::forward(const Eigen::MatrixXf& input)
    {
        if(input.size() != weights.cols())
        {
            std::cout << "\x1B[31minput size doesn't match with weights\x1B[37m\n";
        }

        // std::cout << "input "   << input.rows()   << "x" << input.cols()   << '\n'
        //           << "weights " << weights.rows() << "x" << weights.cols() << '\n'
        //           << "outputs " << outputs.rows() << "x" << outputs.cols() << '\n'
        //           << "=============================================\n";
        // Eigen::VectorXf result = (weights * input) + biases;


 
        outputs = (weights * input).colwise() + biases;
        outputs = outputs.unaryExpr(activation.function);
        // Eigen::VectorXf results(_layerSize_);
        // for(int i = 0 ; i < results.rows() ; i++)
        // {
        //     results[i] = (input.cwiseProduct(weights.row(i).transpose())).sum() + biases[i];
        // }

        // results = results.unaryExpr(activation.function);
        
        // outputs = results;

        return outputs;
    }

    Eigen::MatrixXf Dense::backward(const Eigen::MatrixXf& deltaOutput)
    {
        Eigen::MatrixXf d_pre_activation = deltaOutput.cwiseProduct(outputs.unaryExpr(activation.derivative));

        // Compute gradients for weights and biases
        d_weights = d_pre_activation * input.transpose();
        d_biases = d_pre_activation.rowwise().sum();

        // Compute gradient w.r.t. input for backpropagation
        Eigen::MatrixXf d_input = weights.transpose() * d_pre_activation;       
        
        return d_input;
    }

    void Dense::update_weights(float_t rate)
    {
        this->weights -= this->d_weights * rate;
        this->biases  -= this->d_biases  * rate;
    }

    std::unique_ptr<LayerBase> Dense::createUnique(int layerSize, const char* func)
    {
        return std::make_unique<Dense>(layerSize, func);
    }
}