#include "Dense.hpp"

#include <iostream>

#include "Utility/logs.hpp"

namespace ANN::Models::Layers
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
    
    Eigen::MatrixXf Dense::forward(const Eigen::MatrixXf& input)
    {
        if(input.size() != weights.cols())
        {
            std::cout << "\x1B[31minput size doesn't match with weights\x1B[37m\n";
            std::cout << "input " << Utils::logs::show_matrix_dimensions(input) << ", "
                      << "weights " << Utils::logs::show_matrix_dimensions(weights) << '\n';
        }

        // std::cout << "input "   << input.rows()   << "x" << input.cols()   << '\n'
        //           << "weights " << weights.rows() << "x" << weights.cols() << '\n'
        //           << "outputs " << outputs.rows() << "x" << outputs.cols() << '\n'
        //           << "=============================================\n";
        // Eigen::VectorXf result = (weights * input) + biases;

        this->input = input;
 
        // outputs = (weights * input).colwise() + biases;
        Eigen::MatrixXf weightedSums = (weights * input).colwise() + biases;
        // outputs = outputs.unaryExpr(activation.function);
        outputs = activation.matrixFunction(weightedSums);
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
    //     std::cout << "d_output " << deltaOutput.rows() << "x" << deltaOutput.cols() << "\n";
    //     std::cout << "outputs  " << outputs.rows() << "x" << outputs.cols() << "\n";


        // Eigen::MatrixXf d_pre_activation = deltaOutput.cwiseProduct(outputs.unaryExpr(activation.derivative));
        Eigen::MatrixXf d_pre_activation = deltaOutput.cwiseProduct(activation.matrixDerivative(outputs));
        std::cout << "d_pre_activation:\n" << d_pre_activation << '\n';
        // Compute gradients for weights and biases
        // std::cout << "d_pre " << Utils::logs::show_matrix_dimensions(d_pre_activation) << '\n';
        // std::cout << "input " << Utils::logs::show_matrix_dimensions(input) << '\n';
        d_weights = d_pre_activation * input.transpose();
        std::cout << "d_weights:\n" << d_weights << '\n';
        // std::cout << "2\n";
        d_biases = d_pre_activation.rowwise().sum();
        std::cout << "d_biases:\n" << d_biases << '\n';
        // std::cout << "3\n";
        
        
        // Compute gradient w.r.t. input for backpropagation
        Eigen::MatrixXf d_input = weights.transpose() * d_pre_activation;       
        std::cout << "d_input:\n" << d_biases << '\n';
        // std::cout << "4\n"; 
        
        return d_input;
    }

    void Dense::update_weights(float_t rate)
    {
        // std::cout << "weights " << Utils::logs::show_matrix_dimensions(weights) << '\n';
        std::cout << "weights\n" << weights << '\n';
        // std::cout << "weights grad" << Utils::logs::show_matrix_dimensions(d_weights) << '\n';
        std::cout << "weights grad\n" << d_weights << '\n';
        this->weights -= this->d_weights * rate;
        // std::cout << "biases " << Utils::logs::show_matrix_dimensions(biases) << '\n';
        std::cout << "biases\n" << biases << '\n';
        // std::cout << "biases grad\n" << Utils::logs::show_matrix_dimensions(d_biases) << '\n';
        std::cout << "biases grad\n" << d_biases << '\n';
        this->biases  -= this->d_biases  * rate;
    }

    std::unique_ptr<LayerBase> Dense::createUnique(int layerSize, const char* func)
    {
        return std::make_unique<Dense>(layerSize, func);
    }
}