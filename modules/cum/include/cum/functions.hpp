#pragma once

#include "Core.hpp"

namespace cum::functions
{
    typedef cumeric_t (*function_t)(cumeric_t);
    typedef struct
    {
        const char* name;
        function_t function;
        function_t derivative;
    } activation_t;    

    cumeric_t linear(cumeric_t x);
    struct Linear
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    cumeric_t gelu(cumeric_t x);
    struct GELU
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    cumeric_t relu(cumeric_t x);
    struct ReLU
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    struct Tanh
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    cumeric_t sigmoid(cumeric_t x);
    struct Sigmoid
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    // void activation(cumeric_t* r, const cumeric_t* v, const char* name, std::size_t N);
    // void activationInPlace(cumeric_t* v, const char* name, std::size_t N);

    // return function object due to passed const char*
    void getFunctionByName(activation_t* activation, const char* name);
    void transform(cumeric_t* r, const cumeric_t* v, activation_t& func, std::size_t N);
    void transformInPlace(cumeric_t* v, activation_t& func, std::size_t N);

    // compute gradient of activation function for backpropagation
    void activationDerivative(cumeric_t* r, const cumeric_t* v, activation_t& func, std::size_t N);
    void activationDerivativeInPlace(cumeric_t* v, activation_t& func, std::size_t N);
} // namespace cum::functions
