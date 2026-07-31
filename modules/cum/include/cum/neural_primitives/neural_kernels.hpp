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
    void feed_forward(cumeric_t* Y, const cumeric_t* W, const cumeric_t* X, const cumeric_t* B, std::size_t in_features, std::size_t out_features, functions::function_id activation=functions::function_id::linear);
    void feed_forward_cached_raw(cumeric_t* Y, cumeric_t* R, const cumeric_t* W, const cumeric_t* X, const cumeric_t* B, const std::size_t in_features, const std::size_t out_features, functions::function_id activation);


    void feed_backward();

    // Matrix& binary_cross_entropy(const cum::Matrix& result, const cum::Matrix& target);

    void BCE(cumeric_t* grad, const cumeric_t* P, const cumeric_t* Y, dim_t N);

}