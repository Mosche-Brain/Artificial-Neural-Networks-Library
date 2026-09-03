//
// Created by jaro on 7/19/26.
//

#pragma once

#include "cum/Core.hpp"

#define leaky_relu_alpha (cum::cumeric_t)0.01

namespace cum::functions::linear_units
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t identity(cumeric_t x);

    cumeric_t relu(cumeric_t x);

    cumeric_t leaky_relu(cumeric_t x);

    cumeric_t gelu(cumeric_t x);

    cumeric_t elu(cumeric_t x);

    /* Parallel versions */

    __event__ identity(cumeric_t* r, const cumeric_t* v, dim_t N);
    __event__ identity_in_place(cumeric_t* v, dim_t N);

    __event__ relu(cumeric_t* r, const cumeric_t* v, dim_t N);
    __event__ relu_in_place(cumeric_t* v, dim_t N);

    __event__ leaky_relu(cumeric_t* r, const cumeric_t* v, dim_t N);
    __event__ leaky_relu_in_place(cumeric_t* v, dim_t N);

    __event__ gelu(cumeric_t* r, const cumeric_t* v, dim_t N);
    __event__ gelu_in_place(cumeric_t* v, dim_t N);

    __event__ elu(cumeric_t* r, const cumeric_t* v, dim_t N);
    __event__ elu_in_place(cumeric_t* v, dim_t N);

    /* ========================== Derivatives ========================== */

    /* Scalar versions */

    cumeric_t identity_deriv(cumeric_t x);

    cumeric_t relu_deriv(cumeric_t x);

    cumeric_t leaky_relu_deriv(cumeric_t x);

    cumeric_t gelu_deriv(cumeric_t x);

    cumeric_t elu_deriv(cumeric_t x);

    /* Parallel versions */

    __event__ identity_deriv(cumeric_t* r, const cumeric_t* v, dim_t N);
    __event__ identity_deriv_in_place(cumeric_t* v, dim_t N);

    __event__ relu_deriv(cumeric_t* r, const cumeric_t* v, dim_t N);
    __event__ relu_deriv_in_place(cumeric_t* v, dim_t N);

    __event__ leaky_relu_deriv(cumeric_t* r, const cumeric_t* v, dim_t N);
    __event__ leaky_relu_deriv_in_place(cumeric_t* v, dim_t N);

    __event__ gelu_deriv(cumeric_t* r, const cumeric_t* v, dim_t N);
    __event__ gelu_deriv_in_place(cumeric_t* v, dim_t N);

    __event__ elu_deriv(cumeric_t* r, const cumeric_t* v, dim_t N);
}
