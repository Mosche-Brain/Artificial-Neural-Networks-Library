//
// Created by jaro on 7/19/26.
//

#pragma once

#include "cum/Core.hpp"
#include "cum/functions/function_id.hpp"

// Fused Kernels for deep learning
namespace cum::neural_primitives::neural_kernels
{
    /*
     * W cols - in_features
     * W rows - out features
     * X rows - in features
     * X cols - batch size (currently not supported)
     * B rows - out features
     * Activation - function_id (optional)
     */
    void feed_forward(cumeric_t* Y, const cumeric_t* W, const cumeric_t* X, const cumeric_t* B, const std::size_t in_features, const std::size_t out_features, functions::function_id activation=functions::function_id::linear);

    void feed_backward();

}