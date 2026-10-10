#pragma once

#include "cum/cum.hpp"
#include "cum/functions.hpp"

namespace YANN::Models::Layers
{
    struct LayerData
    {
        cum::Matrix weights;
        cum::Matrix biases;
        cum::functions::activation_t activation;
    };
}