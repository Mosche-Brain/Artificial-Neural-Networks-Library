//
// Created by jaro on 8/2/26.
//

#ifndef YANN_REDUCTIONS_HPP
#define YANN_REDUCTIONS_HPP

#include "cum/Core.hpp"

namespace cum::LinearAlgebra
{
    /* Matrix ---> Vector */

    void rwise_sum(cumeric_t* r, cumeric_t* mat, dim_t m, dim_t n);
    void cwise_sum(cumeric_t* r, cumeric_t* mat, dim_t m, dim_t n);

    /* Vector ---> Scalar */

    void sum(cumeric_t* r, const cumeric_t* v, dim_t n);
    void asum(cumeric_t* r, const cumeric_t* v, dim_t n);
    void mean(cumeric_t* r, const cumeric_t* v, dim_t n);
    void amean(cumeric_t* r, const cumeric_t* v, dim_t n);
}

#endif //YANN_REDUCTIONS_HPP
