#pragma once
#include "Models/Layer.hpp"
#include "Utils/functions.hpp"


namespace Layers
{
    struct INIT_PARAMS
    {
        int size;
        std::function<double(double)> activation;
        std::function<double(double)> derivative;
    };
    
    INIT_PARAMS lambda(std::function<double(double)> f, int size);
    INIT_PARAMS Flatten(int height, int width);
    INIT_PARAMS Tanh(int size);
    INIT_PARAMS Linear(int size);
    INIT_PARAMS Sigmoid(int size);
    INIT_PARAMS RELunit(int size);
    INIT_PARAMS Normalize(int size);
}
