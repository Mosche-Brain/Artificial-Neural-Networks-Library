#include "cum/functions.hpp"
#include "cum/LinearAlgebra.hpp"
#include "cum/memory.hpp"

#include "internal/context.hpp"

#include <oneapi/mkl/vm/buffer.hpp>
#include <sycl/sycl.hpp>

#include <string.h>
#if defined(BUILD_DEBUG_OUTPUT)
    #include <iostream>
#endif

namespace cum::functions
{
    /* Functors Functions */

    cumeric_t Linear::operator()(cumeric_t x) const { return linear_units::identity(x); }
    
    cumeric_t GELU::operator()(cumeric_t x) const { return linear_units::gelu(x); }
     
    cumeric_t ReLU::operator()(cumeric_t x) const { return linear_units::relu(x); }

    cumeric_t Tanh::operator()(cumeric_t x) const { return hyperbolic::tanh(x); }

    cumeric_t Sigmoid::operator()(cumeric_t x) const { return logistic::sigmoid(x); }

    void get_function_by_name(activation_t* activation, const char* name)
    {
        if(strcmp(name, "linear") == 0 || strcmp(name, "identity") == 0)
        {
            *activation = { function_id::linear, linear_units::identity, linear_units::identity_deriv };
        }
        else if(strcmp(name, "relu") == 0)
        {
            *activation = { function_id::relu, linear_units::relu, linear_units::relu_deriv };
        }
        else if(strcmp(name, "leaky_relu") == 0)
        {
            *activation = { function_id::leaky_relu, linear_units::leaky_relu, linear_units::relu_deriv };
        }
        else if(strcmp(name, "gelu") == 0)
        {
            *activation = { function_id::gelu, linear_units::gelu, linear_units::gelu_deriv };
        }
        else if(strcmp(name, "elu") == 0)
        {
            *activation = { function_id::elu, linear_units::elu, linear_units::elu_deriv };
        }
        else if(strcmp(name, "exp") == 0)
        {
            *activation = { function_id::exp, exponential::exp, exponential::exp_deriv };
        }
        else if(strcmp(name, "exp2") == 0)
        {
            *activation = { function_id::exp2, exponential::exp2, exponential::exp2_deriv };
        }
        else if(strcmp(name, "exp10") == 0)
        {
            *activation = { function_id::exp10, exponential::exp10, exponential::exp10_deriv };
        }
        else if(strcmp(name, "log") == 0)
        {
            *activation = { function_id::log, exponential::log, exponential::log_deriv };
        }
        else if(strcmp(name, "log2") == 0)
        {
            *activation = { function_id::log2, exponential::log2, exponential::log2_deriv };
        }
        else if(strcmp(name, "log10") == 0)
        {
            *activation = { function_id::log10, exponential::log10, exponential::log10_deriv };
        }
        else if(strcmp(name, "sin") == 0)
        {
            *activation = { function_id::sin, trigonometric::sin, trigonometric::sin_deriv };
        }
        else if(strcmp(name, "cos") == 0)
        {
            *activation = { function_id::cos, trigonometric::cos, trigonometric::cos_deriv };
        }
        else if(strcmp(name, "tan") == 0)
        {
            *activation = { function_id::tan, trigonometric::tan, trigonometric::tan };
        }
        else if(strcmp(name, "sinh") == 0)
        {
            *activation = { function_id::sinh, hyperbolic::sinh, hyperbolic::sinh_deriv };
        }
        else if(strcmp(name, "cosh") == 0)
        {
            *activation = { function_id::cosh, hyperbolic::cosh, hyperbolic::cosh_deriv };
        }
        else if(strcmp(name, "tanh") == 0)
        {
            *activation = { function_id::tanh, hyperbolic::tanh, hyperbolic::tanh_deriv };
        }
        else if(strcmp(name, "asin") == 0)
        {
            *activation = { function_id::asin, cyclometric::asin, cyclometric::asin };
        }
        else if(strcmp(name, "acos") == 0)
        {
            *activation = { function_id::cosh, cyclometric::acos, cyclometric::acos_deriv };
        }
        else if(strcmp(name, "atan") == 0)
        {
            *activation = { function_id::atan, cyclometric::atan, cyclometric::atan_deriv };
        }
        else if(strcmp(name, "sigmoid") == 0)
        {
            *activation = { function_id::sigmoid, logistic::sigmoid, logistic::sigmoid_deriv };
        }
        else
        {
            *activation = { function_id::undefined, nullptr, nullptr };
        }
    }



    void transformInPlace(cumeric_t* v, activation_t& func, std::size_t N)
    {
        // #pragma omp parallel for
        // for (std::size_t i = 0; i < N; ++i)
        //     v[i] = func.function(v[i]);
    }

    void activationDerivativeInPlace(cumeric_t* v, activation_t& func, std::size_t N)
    {
        #pragma omp parallel for
        for (std::size_t i = 0; i < N; ++i)
            v[i] = func.derivative(v[i]);
    }

}
