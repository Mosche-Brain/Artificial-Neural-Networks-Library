#pragma once

#include "build_config.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <limits>

#include "math_api.hpp"

namespace YANN::Utils
{
    numeric_t linear(math_api::numeric_t x);
    numeric_t linear_derivative (math_api::numeric_t x);

    numeric_t tanh(numeric_t x);
    numeric_t tanh_derivative(numeric_t x);

    numeric_t sigmoid(numeric_t x);
    numeric_t sigmoid_derivative(numeric_t x);

    numeric_t ReLU(numeric_t x);
    numeric_t ReLU_derivative(numeric_t x);

    numeric_t leaky_ReLU(numeric_t x);
    numeric_t leaky_ReLU_derivative(numeric_t x);

    numeric_t GELU(numeric_t x);
    numeric_t GELU_derivative(numeric_t x);

    matrix_t softmax(const matrix_t& x);
    matrix_t softmax_derivative(const matrix_t& x);
}