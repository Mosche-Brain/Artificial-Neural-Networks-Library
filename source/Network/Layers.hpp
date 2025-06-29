#pragma once
#include "Network/Layer.hpp"
#include "functions.hpp"

struct INIT_PARAMS
{
    int size;
    std::function<double(double)> activation;
    std::function<double(double)> derivative;
};

namespace Layers
{

    INIT_PARAMS Flatten(int height, int width);
    // INIT_PARAMS Tanh(int size);
    INIT_PARAMS Linear(int size);
    INIT_PARAMS Sigmoid(int size);
    INIT_PARAMS RELunit(int size);
    // INIT_PARAMS Normalize(int size);
}