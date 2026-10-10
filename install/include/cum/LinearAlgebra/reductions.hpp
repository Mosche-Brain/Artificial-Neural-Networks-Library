//
// Created by jaro on 8/2/26.
//

#ifndef YANN_REDUCTIONS_HPP
#define YANN_REDUCTIONS_HPP

#include "cum/Core.hpp"

namespace cum::LinearAlgebra
{
    /* Matrix ---> Vector */

    __event__ rwise_sum(cumeric_t* r, cumeric_t* mat, dim_t m, dim_t n);
    __event__ cwise_sum(cumeric_t* r, cumeric_t* mat, dim_t m, dim_t n);

    /* Vector ---> Scalar */

    __event__ sum(cumeric_t* r, const cumeric_t* v, dim_t n);
    __event__ asum(cumeric_t* r, const cumeric_t* v, dim_t n);
    __event__ mean(cumeric_t* r, const cumeric_t* v, dim_t n);
    __event__ amean(cumeric_t* r, const cumeric_t* v, dim_t n);
}

#endif //YANN_REDUCTIONS_HPP
