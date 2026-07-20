#pragma once

#include "functions/trigonometric.hpp"
#include "functions/cyclometric.hpp"
#include "functions/hyperbolic.hpp"
#include "functions/exponential.hpp"
#include "functions/logistic.hpp"
#include "functions/linear_units.hpp"
#include "functions/various.hpp"

#include "functions/function_id.hpp"

namespace cum::functions
{



    struct Linear
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    struct GELU
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    struct ReLU
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    struct leaky_ReLU
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    struct Tanh
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    struct Sigmoid
    {
        cumeric_t operator()(cumeric_t x) const;
    };




    // return function object due to passed const char*
    void getFunctionByName(activation_t* activation, const char* name);
    void transformInPlace(cumeric_t* v, activation_t& func, std::size_t N);

    // compute gradient of activation function for backpropagation
    void activationDerivative(cumeric_t* r, const cumeric_t* v, activation_t& func, std::size_t N);
    void activationDerivativeInPlace(cumeric_t* v, activation_t& func, std::size_t N);
} // namespace cum::functions
