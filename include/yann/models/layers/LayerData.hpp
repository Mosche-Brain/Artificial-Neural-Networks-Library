/**
 * @file LayerData.hpp
 * @brief Legacy structure holding layer parameters (weights, biases, and activation function).
 */

#pragma once

#include "cum/cum.hpp"
#include "cum/functions.hpp"

namespace YANN::Models::Layers
{
    /**
     * @struct LayerData
     * @brief Storage struct encapsulating weight/bias matrices and activation function pointer.
     */
    struct LayerData
    {
        cum::Matrix weights;                     /**< Layer weight matrix. */
        cum::Matrix biases;                      /**< Layer bias matrix. */
        cum::functions::activation_t activation; /**< Layer activation function. */
    };
}
