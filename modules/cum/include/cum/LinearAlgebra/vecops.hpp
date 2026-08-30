//
// Created by jaro on 8/30/26.
//

#ifndef YANN_VECOPS_HPP
#define YANN_VECOPS_HPP

#include "cum/Core.hpp"

#include "BLAS/level1.hpp"

namespace cum::LinearAlgebra
{
    void vadd(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, dim_t n);

    void vadd(cumeric_t* v, const cumeric_t* u, dim_t n);

    void vsub(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, dim_t n);

    void vsub(cumeric_t* v, const cumeric_t* u, dim_t n);

    void vmul(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, dim_t n);

    void vmul(cumeric_t* v, const cumeric_t* u, dim_t n);

    void vdiv(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, dim_t n);

    void vdiv(cumeric_t* v, const cumeric_t* u, dim_t n);
}


#endif //YANN_VECOPS_HPP
