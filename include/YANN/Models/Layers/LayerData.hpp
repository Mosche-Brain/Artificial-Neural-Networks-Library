#pragma once

#include "math_api.hpp"
#include "Utility/Activation.hpp"

namespace YANN::Models::Layers
{
    struct LayerData
    {
        matrix_t weights;
        matrix_t biases;
        Utils::activation_t activation;
    };
}