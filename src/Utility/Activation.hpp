#pragma once

#include "build_config.hpp"
#include <Eigen/Dense>
#include <functional>
#include <map>

#include "functions.hpp"

namespace YANN::Utils
{
    class Activation
    {
    public:
        Activation();

        Activation(std::function<numeric_t(numeric_t)> function, 
                   std::function<numeric_t(numeric_t)> derivative);

        Activation(std::function<matrix_t(matrix_t)> matrixFunction,
                   std::function<matrix_t(matrix_t)> matrixDerivative);

        Activation(const char* func);


        std::function<numeric_t(numeric_t)> function;
        std::function<numeric_t(numeric_t)> derivative;

        std::function<matrix_t(matrix_t)> matrixFunction;
        std::function<matrix_t(matrix_t)> matrixDerivative;
    };
    
}