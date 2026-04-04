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

    activation_t getActivationByName(const char* name)
    {
        if (std::strcmp(name, "linear") == 0)
        {
            return {"linear", linear, linear_derivative, 
                    nullptr,
                    nullptr };
        }
        else if (std::strcmp(name, "tanh") == 0)
        {
            return {"tanh", tanh, tanh_derivative,
                    nullptr,
                    nullptr };
        }
        else if (std::strcmp(name, "sigmoid") == 0)
        {
            return {"sigmoid", sigmoid, sigmoid_derivative,
                    nullptr,
                    nullptr };
        }
        else if (std::strcmp(name, "relu") == 0)
        {
            return {"relu", ReLU, ReLU_derivative,
                    nullptr,
                    nullptr };
        }
        else if (std::strcmp(name, "leaky_relu") == 0)
        {
            return {"leaky_relu", leaky_ReLU, leaky_ReLU_derivative,
                    nullptr,
                    nullptr };
        }
        else if (std::strcmp(name, "gelu") == 0)
        {
            return {"gelu", GELU, GELU_derivative,
                    nullptr,
                    nullptr };
        }
        else if (std::strcmp(name, "softmax") == 0)
        {
            // return {"softmax", nullptr, nullptr, softmax, softmax_derivative};
            return {"softmax", nullptr, nullptr, nullptr, nullptr};
        }
        else
        {
            throw std::invalid_argument(std::string("Unknown activation function: ") + name);
        }
    }
}