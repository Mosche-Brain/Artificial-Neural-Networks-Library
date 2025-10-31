#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <limits>

namespace ANN::Utils
{
    float_t linear(float_t x);
    float_t linear_derivative (float_t x);

    float_t tanh(float_t x);
    float_t tanh_derivative(float_t x);

    float_t sigmoid(float_t x);
    float_t sigmoid_derivative(float_t x);

    float_t ReLU(float_t x);
    float_t ReLU_derivative(float_t x);

    float_t GELU(float_t x);
    float_t GELU_derivative(float_t x);

    Eigen::MatrixXf softmax(const Eigen::MatrixXf& x);
    Eigen::MatrixXf softmax_derivative(const Eigen::MatrixXf& x);
}