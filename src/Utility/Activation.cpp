#include "Activation.hpp"

#include <cum/functions.hpp>
#include <cstring>

namespace YANN::Utils
{

    // activation_t getActivationByName(const char* name)
    // {
    //     if (std::strcmp(name, "linear") == 0)
    //     {
    //         return {"linear", linear, linear_derivative, 
    //                 nullptr,
    //                 nullptr };
    //     }
    //     else if (std::strcmp(name, "tanh") == 0)
    //     {
    //         return {"tanh", tanh, tanh_derivative,
    //                 nullptr,
    //                 nullptr };
    //     }
    //     else if (std::strcmp(name, "sigmoid") == 0)
    //     {
    //         return {"sigmoid", sigmoid, sigmoid_derivative,
    //                 nullptr,
    //                 nullptr };
    //     }
    //     else if (std::strcmp(name, "relu") == 0)
    //     {
    //         return {"relu", ReLU, ReLU_derivative,
    //                 nullptr,
    //                 nullptr };
    //     }
    //     else if (std::strcmp(name, "leaky_relu") == 0)
    //     {
    //         return {"leaky_relu", leaky_ReLU, leaky_ReLU_derivative,
    //                 nullptr,
    //                 nullptr };
    //     }
    //     else if (std::strcmp(name, "gelu") == 0)
    //     {
    //         return {"gelu", GELU, GELU_derivative,
    //                 nullptr,
    //                 nullptr };
    //     }
    //     else if (std::strcmp(name, "softmax") == 0)
    //     {
    //         // return {"softmax", nullptr, nullptr, softmax, softmax_derivative};
    //         return {"softmax", nullptr, nullptr, nullptr, nullptr};
    //     }
    //     else
    //     {
    //         throw std::invalid_argument(std::string("Unknown activation function: ") + name);
    //     }
    // }
}