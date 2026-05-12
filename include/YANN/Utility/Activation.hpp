#pragma once

#include <functional>
#include <map>

#include "cum/Matrix.hpp"
#include "cum/LinearAlgebra.hpp"
#include "functions.hpp"

namespace YANN::Utils
{
    class Activation
    {
    public:
        Activation();

        Activation(std::function<cum::cumeric_t(cum::cumeric_t)> function, 
                   std::function<cum::cumeric_t(cum::cumeric_t)> derivative);

        Activation(std::function<cum::Matrix(cum::Matrix)> matrixFunction,
                   std::function<cum::Matrix(cum::Matrix)> matrixDerivative);

        Activation(const char* func);


        std::function<cum::cumeric_t(cum::cumeric_t)> function;
        std::function<cum::cumeric_t(cum::cumeric_t)> derivative;

        std::function<cum::Matrix(cum::Matrix)> matrixFunction;
        std::function<cum::Matrix(cum::Matrix)> matrixDerivative;
    };

    struct activation_t
    {
        const char* name;
        cum::cumeric_t (*function)(cum::cumeric_t);
        cum::cumeric_t (*derivative)(cum::cumeric_t);

        cum::Matrix (*matFunction)(cum::Matrix);
        cum::Matrix (*matDerivative)(cum::Matrix);
    };

    activation_t getActivationByName(const char* name);
    
}