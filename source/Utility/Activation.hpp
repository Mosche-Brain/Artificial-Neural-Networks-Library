#pragma once

#include <Eigen/Dense>

#include <functional>
#include <map>

namespace SNN::Utils
{
    struct Activation
    {
        std::function<float_t(float_t)> function;
        std::function<float_t(float_t)> derivative;
    };
    
    

    Activation stringToFunction(const char* func);
}