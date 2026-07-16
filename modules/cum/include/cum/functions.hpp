#pragma once

#include "Core.hpp"

namespace cum::functions
{
    enum class ActivationFunction
    {
        undefined = 0,
        linear,
        gelu,
        relu,
        tanh,
        sigmoid
    };
    // typedef cumeric_t (*function_t)(cumeric_t);
    typedef cumeric_t (*function_t)(cumeric_t);
    typedef struct
    {
        ActivationFunction name;
        function_t function;
        function_t derivative;
    } activation_t;    


    void fill(cumeric_t* v, cumeric_t val, const std::size_t N);

    cumeric_t linear(cumeric_t x);
    struct Linear
    {
        cumeric_t operator()(cumeric_t x) const;
    };
    void linear(cumeric_t* r, const cumeric_t* v, const std::size_t N);
    void linearInPlace(cumeric_t* v, const std::size_t N);


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
    void relu(cumeric_t* r, const cumeric_t* v, const std::size_t N);
    void reluInPlace(cumeric_t* v, const std::size_t N);
    void relu_derivative(cumeric_t* r, const cumeric_t* v, const std::size_t N);


    cumeric_t tanh(cumeric_t x);
    cumeric_t tanh_derivative(cumeric_t x);
    void tanh(cumeric_t* r, const cumeric_t* v, const std::size_t N);
    void tanhInPlace(cumeric_t* v, const std::size_t N);

    struct Tanh
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    cumeric_t sigmoid(cumeric_t x);
    cumeric_t sigmoid_derivative(cumeric_t x);
    struct Sigmoid
    {
        cumeric_t operator()(cumeric_t x) const;
    };
    void sigmoid(cumeric_t* r, const cumeric_t* v, const std::size_t N);
    void sigmoidInPlace(cumeric_t* v, const std::size_t N);
    void sigmoid_derivative(cumeric_t* r, const cumeric_t* v, const std::size_t N);


    cumeric_t min(const cumeric_t a, const cumeric_t b);
    cumeric_t max(const cumeric_t a, const cumeric_t b);

    void clip(cumeric_t* r, const cumeric_t* v, const cumeric_t min, const cumeric_t max, const std::size_t N);
    void clipInPlace(cumeric_t* v, const cumeric_t min, const cumeric_t max, const std::size_t N);
    // void clip(cummulative_t* r, const cumeric_t* v, const cumeric_t min, const cumeric_t max, const std::size_t N);
    // void clipInPlace(cummulative_t* v, const cummulative_t min, const cummulative_t max, const std::size_t N);
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
