#include "cum/functions.hpp"

#include <math.h>
#include <string.h>

#if defined(BUILD_DEBUG_OUTPUT)
    #include <iostream>
#endif

namespace cum::functions
{
    cumeric_t linear(cumeric_t x) { return x; }
    cumeric_t linear_derivative(cumeric_t x) { return 1; }
    cumeric_t Linear::operator()(cumeric_t x) const
    {
        return linear(x);
    }
    
    cumeric_t gelu(cumeric_t x)
    {
        #if defined(BUILD_USE_F64)
        #define M_SQRT_2_OVER_PI 0.7978845608028654
        #define M_SQRT_2_OVER_PI_A 0.044715000000000000
        #elif defined(BUILD_USE_F32)
        #define M_SQRT_2_OVER_PI 0.79788456f
        #define M_SQRT_2_OVER_PI_A 0.044715f
        #elif defined(BUILD_USE_F16)
        #define M_SQRT_2_OVER_PI 0.79788456_c
        #define M_SQRT_2_OVER_PI_A 0.044715_c
        #elif defined(BUILD_USE_BF16)
        #define M_SQRT_2_OVER_PI 0.79788456bf16
        #define M_SQRT_2_OVER_PI_A 0.044715bf16
        #else
        #define M_SQRT_2_OVER_PI 0.79788456f
        #define M_SQRT_2_OVER_PI_A 0.044715f
        #endif

        return 0.5_c * x * (1.0_c + tanhf(M_SQRT_2_OVER_PI * (x + M_SQRT_2_OVER_PI_A * x * x * x)));
    }
    cumeric_t GELU_derivative(cumeric_t x)
    {
        #if defined(BUILD_USE_F64)
        #define M_SQRT_2_OVER_PI 0.7978845608028654
        #define M_SQRT_2_OVER_PI_A 0.044715000000000000
        #elif defined(BUILD_USE_F32)
        #define M_SQRT_2_OVER_PI 0.79788456f
        #define M_SQRT_2_OVER_PI_A 0.044715f
        #elif defined(BUILD_USE_F16)
        #define M_SQRT_2_OVER_PI 0.79788456_c
        #define M_SQRT_2_OVER_PI_A 0.044715_c
        #elif defined(BUILD_USE_BF16)
        #define M_SQRT_2_OVER_PI 0.79788456bf16
        #define M_SQRT_2_OVER_PI_A 0.044715bf16
        #else
        #define M_SQRT_2_OVER_PI 0.79788456f
        #define M_SQRT_2_OVER_PI_A 0.044715f
        #endif

        cumeric_t tanh_arg = M_SQRT_2_OVER_PI * (x + M_SQRT_2_OVER_PI_A * x * x * x);
        cumeric_t tanh_val = tanhf(tanh_arg);
        cumeric_t sech_squared = 1 - tanh_val * tanh_val;
        return 0.5_c * (1.0_c + tanh_val) + (M_SQRT_2_OVER_PI * (x + 3 * M_SQRT_2_OVER_PI_A * x * x) * sech_squared) / 6.0_c;
    }
    cumeric_t GELU::operator()(cumeric_t x) const
    {
        return gelu(x);
    }
    
    cumeric_t relu(cumeric_t x) { return x > 0.0_c ? x : 0.0_c; }
    cumeric_t relu_derivative(cumeric_t x) { return x > 0.0_c ? 1.0_c : 0.0_c; }
    cumeric_t ReLU::operator()(cumeric_t x) const
    {
        return relu(x);
    }

    cumeric_t tanh(cumeric_t x) { return tanhf(x); }
    cumeric_t tanh_derivative(cumeric_t x) { return 1.0_c - x * x; }
    cumeric_t Tanh::operator()(cumeric_t x) const
    {
        return tanhf(x);
    }

    cumeric_t sigmoid(cumeric_t x) { return 1.0_c / (1.0_c + expf(-x)); }
    cumeric_t sigmoid_derivative(cumeric_t x) { return x * (1.0_c - x); }
    cumeric_t Sigmoid::operator()(cumeric_t x) const
    {
        return sigmoid(x);
    }

    void activation(cumeric_t* r, const cumeric_t* v, const char* name, std::size_t N)
    {
        if (strcmp(name, "linear") == 0)
        {
            auto func = Linear();
            for (std::size_t i = 0; i < N; ++i)
                r[i] = func(v[i]);
        }
        else if (strcmp(name, "relu") == 0)
        {
            auto func = ReLU();
            for (std::size_t i = 0; i < N; ++i)
                r[i] = func(v[i]);
        }
        else if (strcmp(name, "gelu") == 0)
        {
            auto func = GELU();
            for (std::size_t i = 0; i < N; ++i)
                r[i] = func(v[i]);
        }
        else if (strcmp(name, "tanh") == 0)
        {
            auto func = Tanh();
            for (std::size_t i = 0; i < N; ++i)
                r[i] = func(v[i]);
        }
        else if (strcmp(name, "sigmoid") == 0)
        {
            auto func = Sigmoid();
            for (std::size_t i = 0; i < N; ++i)
                r[i] = func(v[i]);
        }
    }

    void getFunctionByName(activation_t* activation, const char* name)
    {
        if(strcmp(name, "linear") == 0)
        {
            activation = new activation_t{ ActivationFunction::linear, linear, linear_derivative };
        }
        else if(strcmp(name, "gelu") == 0)
        {
            *activation = { ActivationFunction::gelu, gelu, GELU_derivative };
        }
        else if(strcmp(name, "relu") == 0)
        {
            *activation = { ActivationFunction::relu, relu, relu_derivative };
        }
        else if(strcmp(name, "tanh") == 0)
        {
            *activation = { ActivationFunction::tanh, tanh, tanh_derivative };
        }
        else if(strcmp(name, "sigmoid") == 0)
        {
            *activation = { ActivationFunction::sigmoid, sigmoid, sigmoid_derivative };
        }
        else
        {
            *activation = { ActivationFunction::undefined, nullptr, nullptr };
        }
    }

    void transform(cumeric_t* r, const cumeric_t* v, activation_t& func, std::size_t N)
    {
        #pragma omp parallel for
        for (std::size_t i = 0; i < N; ++i)
            r[i] = func.function(v[i]);
    }

    void transformInPlace(cumeric_t* v, activation_t& func, std::size_t N)
    {
        #pragma omp parallel for
        for (std::size_t i = 0; i < N; ++i)
            v[i] = func.function(v[i]);
    }

    void activationDerivative(cumeric_t* r, const cumeric_t* v, activation_t& func, std::size_t N)
    {
        #pragma omp parallel for
        for (std::size_t i = 0; i < N; ++i)
            r[i] = func.derivative(v[i]);
    }

    void activationDerivativeInPlace(cumeric_t* v, activation_t& func, std::size_t N)
    {
        #pragma omp parallel for
        for (std::size_t i = 0; i < N; ++i)
            v[i] = func.derivative(v[i]);
    }

} // namespace cum::functions
