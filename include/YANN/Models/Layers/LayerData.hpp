#pragma once

#include "cum/cum.hpp"
#include "Utility/Activation.hpp"

namespace YANN::Models::Layers
{
    struct LayerData
    {
        cum::Matrix weights;
        cum::Matrix biases;
        Utils::activation_t activation;
    };
}