#include "Activation.hpp"

namespace YANN::Utils
{
    Activation::Activation()
    {
        
    }

    Activation::Activation(const char* func_name)
    {
        if (std::strcmp(func_name, "linear") == 0)
        {
            function = linear;
            derivative = linear_derivative;
            matrixFunction = [this](const matrix_t& X) { return X.unaryExpr(&linear); };
            matrixDerivative = [this](const matrix_t& X) { return X.unaryExpr(&linear_derivative); };
        }
        else if (std::strcmp(func_name, "tanh") == 0)
        {
            function = tanh;
            derivative = tanh_derivative;
            matrixFunction = [this](const matrix_t& X) { return X.unaryExpr(&tanh); };
            matrixDerivative = [this](const matrix_t& X) { return X.unaryExpr(&tanh_derivative); };
        }
        else if (std::strcmp(func_name, "sigmoid") == 0)
        {
            function = sigmoid;
            derivative = sigmoid_derivative;
            matrixFunction = [this](const matrix_t& X) { return X.unaryExpr(&sigmoid); };
            matrixDerivative = [this](const matrix_t& X) { return X.unaryExpr(&sigmoid_derivative); };
        }
        else if (std::strcmp(func_name, "relu") == 0)
        {
            function = ReLU;
            derivative = ReLU_derivative;
            matrixFunction = [this](const matrix_t& X) { return X.unaryExpr(function); };
            matrixDerivative = [this](const matrix_t& X) { return X.unaryExpr(derivative); };
        }
        else if (std::strcmp(func_name, "leaky_relu") == 0)
        {
            function = leaky_ReLU;
            derivative = leaky_ReLU_derivative;
            matrixFunction = [this](const matrix_t& X) { return X.unaryExpr(&leaky_ReLU); };
            matrixDerivative = [this](const matrix_t& X) { return X.unaryExpr(&leaky_ReLU_derivative); };
        }
        else if (std::strcmp(func_name, "gelu") == 0)
        {
            function = GELU;
            derivative = GELU_derivative;
            matrixFunction = [this](const matrix_t& X) { return X.unaryExpr(function); };
            matrixDerivative = [this](const matrix_t& X) { return X.unaryExpr(derivative); };
        }
        else if (std::strcmp(func_name, "softmax") == 0)
        {
            matrixFunction = softmax;
            matrixDerivative = softmax_derivative;

            function = nullptr;
            derivative = nullptr;
        }
        else
        {
            throw std::invalid_argument(std::string("Unknown activation function: ") + func_name);
        }
    }
}