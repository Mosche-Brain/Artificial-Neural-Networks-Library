#pragma once

#include <Eigen/Dense>

#include <functional>
#include <map>

#include "functions.hpp"

namespace ANN::Utils
{
    class Activation
    {
    public:
        Activation();

        Activation(std::function<float_t(float_t)> function, 
                   std::function<float_t(float_t)> derivative);

        Activation(std::function<Eigen::MatrixXf(Eigen::MatrixXf)> matrixFunction,
                   std::function<Eigen::MatrixXf(Eigen::MatrixXf)> matrixDerivative);

        Activation(const char* func);


        std::function<float_t(float_t)> function;
        std::function<float_t(float_t)> derivative;

        std::function<Eigen::MatrixXf(Eigen::MatrixXf)> matrixFunction;
        std::function<Eigen::MatrixXf(Eigen::MatrixXf)> matrixDerivative;
    };
    
}