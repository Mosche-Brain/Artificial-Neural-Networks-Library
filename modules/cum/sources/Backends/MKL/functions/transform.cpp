/* Created by jaro on 7/20/26. */
#include <cstdint>

#include "cum/functions/linear_units.hpp"
#include "cum/functions/trigonometric.hpp"
#include "cum/functions/cyclometric.hpp"
#include "cum/functions/hyperbolic.hpp"
#include "cum/functions/exponential.hpp"
#include "cum/functions/logistic.hpp"
#include "cum/functions/linear_units.hpp"
#include "cum/functions/various.hpp"

#include "cum/functions/transform.hpp"

namespace cum::functions
{
    void transform(cumeric_t* r, const cumeric_t* v, const std::size_t N, const activation_t& activation)
    {
        if (activation.name != function_id::undefined)
        {
            transform(r, v, N, activation.name);
        }
        else if (activation.function != nullptr)
        {
            #pragma omp parallel for
            for (std::size_t i = 0; i < N; ++i)
                r[i] = activation.function(v[i]);
        }
        else
        {
            throw std::runtime_error("Unknown activation type (transform), used " + std::to_string(static_cast<unsigned char>(activation.name)));
        }
    }

    void transform_deriv(cumeric_t* r, const cumeric_t* v, const std::size_t N, const activation_t& activation)
    {
        if (activation.name != function_id::undefined)
        {
            transform_deriv(r, v, N, activation.name);
        }
        else if (activation.derivative != nullptr)
        {
            #pragma omp parallel for
            for (std::size_t i = 0; i < N; ++i)
                r[i] = activation.derivative(v[i]);
        }
        else
        {
            throw std::runtime_error("Unknown activation type (transform_deriv)");
        }
    }

    void transform(cumeric_t* r, const cumeric_t* v, const std::size_t N, const function_id activation)
    {
        switch (activation)
        {
            case function_id::linear:
            {
                linear_units::identity(r, v, N);
                break;
            }
            case function_id::relu:
            {
                linear_units::relu(r, v, N);
                break;
            }
            case function_id::leaky_relu:
            {
                linear_units::leaky_relu(r, v, N);
                break;
            }
            case function_id::gelu:
            {
                linear_units::gelu(r, v, N);
                break;
            }
            case function_id::elu:
            {
                linear_units::elu(r, v, N);
                break;
            }
            case function_id::exp:
            {
                exponential::exp(r, v, N);
                break;
            }
            case function_id::exp2:
            {
                exponential::exp2(r, v, N);
                break;
            }
            case function_id::exp10:
            {
                exponential::exp10(r, v, N);
                break;
            }
            case function_id::log:
            {
                exponential::log(r, v, N);
                break;
            }
            case function_id::log2:
            {
                exponential::log2(r, v, N);
                break;
            }
            case function_id::log10:
            {
                exponential::log10(r, v, N);
                break;
            }
            case function_id::sigmoid:
            {
                logistic::sigmoid(r, v, N);
                break;
            }
            case function_id::sin:
            {
                trigonometric::sin(r, v, N);
                break;
            }
            case function_id::cos:
            {
                trigonometric::cos(r, v, N);
                break;
            }
            case function_id::tan:
            {
                trigonometric::tan(r, v, N);
                break;
            }
            case function_id::sinh:
            {
                hyperbolic::sinh(r, v, N);
                break;
            }
            case function_id::cosh:
            {
                hyperbolic::cosh(r, v, N);
                break;
            }
            case function_id::tanh:
            {
                hyperbolic::tanh(r, v, N);
                break;
            }
            case function_id::asin:
            {
                cyclometric::asin(r, v, N);
                break;
            }
            case function_id::acos:
            {
                cyclometric::acos(r, v, N);
                break;
            }
            case function_id::atan:
            {
                cyclometric::atan(r, v, N);
                break;
            }
            default:
            {
                throw std::runtime_error("Unknown activation function (transform(cumeric_t*, const cumeric_t*, std::size_t, function_id))");

                // if (func.derivative == nullptr)
                //     break;
                //
                // #pragma omp parallel for
                // for (std::int64_t i = 0; i < N; ++i)
                //     r[i] = func.function(v[i]);
                // std::cerr << "Unknown activation function" << std::endl;
            }
        }
    }

    void transform_deriv(cumeric_t* r, const cumeric_t* v, const std::size_t N, const function_id activation)
    {
        switch (activation)
        {
            case function_id::linear:
            {
                linear_units::identity_deriv(r, v, N);
                break;
            }
            case function_id::relu:
            {
                linear_units::relu_deriv(r, v, N);
                break;
            }
            case function_id::leaky_relu:
            {
                linear_units::leaky_relu_deriv(r, v, N);
                break;
            }
            case function_id::gelu:
            {
                linear_units::gelu_deriv(r, v, N);
                break;
            }
            case function_id::elu:
            {
                linear_units::elu_deriv(r, v, N);
                break;
            }
            case function_id::exp:
            {
                exponential::exp_deriv(r, v, N);
                break;
            }
            case function_id::exp2:
            {
                exponential::exp2_deriv(r, v, N);
                break;
            }
            case function_id::exp10:
            {
                exponential::exp10_deriv(r, v, N);
                break;
            }
            case function_id::log:
            {
                exponential::log_deriv(r, v, N);
                break;
            }
            case function_id::log2:
            {
                exponential::log2_deriv(r, v, N);
                break;
            }
            case function_id::log10:
            {
                exponential::log10_deriv(r, v, N);
                break;
            }
            case function_id::sigmoid:
            {
                logistic::sigmoid_deriv(r, v, N);
                break;
            }
            case function_id::sin:
            {
                trigonometric::sin_deriv(r, v, N);
                break;
            }
            case function_id::cos:
            {
                trigonometric::cos_deriv(r, v, N);
                break;
            }
            case function_id::tan:
            {
                trigonometric::tan_deriv(r, v, N);
                break;
            }
            case function_id::sinh:
            {
                hyperbolic::sinh_deriv(r, v, N);
                break;
            }
            case function_id::cosh:
            {
                hyperbolic::cosh_deriv(r, v, N);
                break;
            }
            case function_id::tanh:
            {
                hyperbolic::tanh_deriv(r, v, N);
                break;
            }
            case function_id::asin:
            {
                cyclometric::asin_deriv(r, v, N);
                break;
            }
            case function_id::acos:
            {
                cyclometric::acos_deriv(r, v, N);
                break;
            }
            case function_id::atan:
            {
                cyclometric::atan_deriv(r, v, N);
                break;
            }
            default:
            {
                throw std::runtime_error("Unknown activation function (in transform_deriv(cumeric_t*, const cumeric_t*, std::size_t, function_id))");
                // Apply with naive loop
                // if (activation == nullptr)
                //     break;

                // #pragma omp parallel for
                // for (std::size_t i = 0; i < N; ++i)
                    // r[i] = func.derivative(v[i]);
                // std::cerr << "Unknown activation function" << std::endl;
            }
        }
    }


}
