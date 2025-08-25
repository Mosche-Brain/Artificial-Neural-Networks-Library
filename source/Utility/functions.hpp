#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace SNN::Utils
{
    float_t tanhf_derivative(float_t x);

    float_t sigmoid(float_t x);
    float_t sigmoid_derivative(float_t x);

    float_t ReLU(float_t x);
    float_t ReLU_derivative(float_t x);

    Eigen::VectorXf softmax(Eigen::VectorXd x);
    Eigen::VectorXf softmax_derivative(Eigen::VectorXd x);

}