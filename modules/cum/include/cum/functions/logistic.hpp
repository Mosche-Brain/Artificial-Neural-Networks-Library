//
// Created by jaro on 7/19/26.
//

#pragma once

#include "cum/Core.hpp"

namespace cum::functions::logistic
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t sigmoid(cumeric_t x);

    /* Parallel versions */

    void sigmoid(cumeric_t* r, const cumeric_t* v, const std::size_t N);
    void sigmoid_in_place(cumeric_t* v, const std::size_t N);

    void softmax(cumeric_t* r, const cumeric_t* v, const dim_t N);
    void softmax_in_place(cumeric_t* v, const dim_t N);

    /* ========================== Derivatives ========================== */

    /* Scalar versions */

    cumeric_t sigmoid_deriv(cumeric_t x);
    cumeric_t sigmoid_deriv_from_result(cumeric_t x);

    /* Parallel versions */

    void sigmoid_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void sigmoid_deriv_from_result(cumeric_t* r, const cumeric_t* v, const std::size_t N);
    void sigmoid_deriv_in_place(cumeric_t* v, const std::size_t N);
    void sigmoid_deriv_from_result_in_place(cumeric_t* v, std::size_t N);

    void softmax_deriv(cumeric_t* r, const cumeric_t* v, dim_t N);
    void softmax_deriv_in_place(cumeric_t* v, dim_t N);
    void softmax_deriv_from_result(cumeric_t* r, const cumeric_t* v, const std::size_t N);

}