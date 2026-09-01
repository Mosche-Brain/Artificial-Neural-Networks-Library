//
// Created by jaro on 8/30/26.
//

#ifndef YANN_VECOPS_HPP
#define YANN_VECOPS_HPP

#include "cum/Core.hpp"

#include "BLAS/level1.hpp"

namespace cum::LinearAlgebra
{
    __event__ vadd(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, dim_t n);

    __event__ vadd(cumeric_t* v, const cumeric_t* u, dim_t n);

    __event__ vsub(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, dim_t n);

    __event__ vsub(cumeric_t* v, const cumeric_t* u, dim_t n);

    __event__ vmul(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, dim_t n);

    __event__ vmul(cumeric_t* v, const cumeric_t* u, dim_t n);

    __event__ vdiv(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, dim_t n);

    __event__ vdiv(cumeric_t* v, const cumeric_t* u, dim_t n);
}


#endif //YANN_VECOPS_HPP
