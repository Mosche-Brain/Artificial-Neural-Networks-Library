#include "Activation.hpp"

namespace ANN::Utils
{
    Activation::Activation()
    {
        
    }

    Activation::Activation(const char* func)
    {
        // std::map<const char*, Activation> map = 
        // {
        //     {"tanh", {Utils::tanh, Utils::tanh_derivative }},
        //     {"sigmoid", {Utils::sigmoid, Utils::sigmoid_derivative }},
        //     {"relu", {Utils::ReLU, Utils::ReLU_derivative }},
        // };

        if(strcmp(func, "tanh"))
        {
            function = tanh;
            derivative = tanh_derivative;

            matrixFunction = [this](Eigen::MatrixXf X) -> Eigen::MatrixXf 
            {
                return X.unaryExpr(function);
            };

            matrixDerivative = [this](Eigen::MatrixXf X) -> Eigen::MatrixXf 
            {
                return X.unaryExpr(derivative);
            };
        }
        else if(strcmp(func, "sigmoid"))
        {
            function = sigmoid;
            derivative = sigmoid_derivative;
    
            matrixFunction = [this](Eigen::MatrixXf X) -> Eigen::MatrixXf 
            {
                return X.unaryExpr(function);
            };
    
            matrixDerivative = [this](Eigen::MatrixXf X) -> Eigen::MatrixXf 
            {
                return X.unaryExpr(derivative);
            };
        }
        else if(strcmp(func, "relu"))
        {
            function = ReLU;
            derivative = ReLU_derivative;
    
            matrixFunction = [this](Eigen::MatrixXf X) -> Eigen::MatrixXf 
            {
                return X.unaryExpr(function);
            };
    
            matrixDerivative = [this](Eigen::MatrixXf X) -> Eigen::MatrixXf 
            {
                return X.unaryExpr(derivative);
            };
        }
        else if(strcmp(func, "gelu"))
        {
            function = GELU;
            derivative = GELU_derivative;
    
            matrixFunction = [this](Eigen::MatrixXf X) -> Eigen::MatrixXf 
            {
                return X.unaryExpr(function);
            };
    
            matrixDerivative = [this](Eigen::MatrixXf X) -> Eigen::MatrixXf 
            {
                return X.unaryExpr(derivative);
            };
        }
        else if(strcmp(func, "relu"))
        {
            // function = ReLU;
            // derivative = ReLU_derivative;
    
            matrixFunction = softmax;
    
            matrixDerivative = softmax_derivative;
        }
    }
}