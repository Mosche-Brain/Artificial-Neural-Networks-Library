#pragma once

#include <functional>
#include <map>

#include "math_api.hpp"
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

    struct activation_t
    {
        const char* name;
        numeric_t (*function)(numeric_t);
        numeric_t (*derivative)(numeric_t);

        matrix_t (*matFunction)(matrix_t);
        matrix_t (*matDerivative)(matrix_t);
    };

    activation_t getActivationByName(const char* name);
    
}