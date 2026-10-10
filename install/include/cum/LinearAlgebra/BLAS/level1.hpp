//
// Created by jaro on 8/2/26.
//

#ifndef YANN_LEVEL1_HPP
#define YANN_LEVEL1_HPP

#include "cum/Core.hpp"

namespace cum::blas
{
    /* vector-scalar ops */

    __event__ axpy(dim_t n, cumeric_t alpha, const cumeric_t* x, dim_t incx, cumeric_t* y, dim_t incy);
    __event__ axpy(dim_t n, cumeric_t alpha, const cumeric_t* x, dim_t incx, cumeric_t beta, cumeric_t* y, dim_t incy);

    /* vector-vector ops */

    __event__ dot(dim_t n, const cumeric_t* x, dim_t incx, const cumeric_t* y, dim_t incy, cumeric_t* result);

    /* vector to scalar reductions */

    __event__ asum(dim_t n, const cumeric_t* x, dim_t incx, cumeric_t* result);
    __event__ nrm2(dim_t n, const cumeric_t* x, dim_t incx, cumeric_t* result);

    __event__ iamax(dim_t n, const cumeric_t* x, dim_t incx, dim_t* result);
    __event__ iamin(dim_t n, const cumeric_t* x, dim_t incx, dim_t* result);

    __event__ amax(dim_t n, const cumeric_t* x, dim_t incx, cumeric_t* result);
    __event__ amin(dim_t n, const cumeric_t* x, dim_t incx, cumeric_t* result);

	__event__ scal(dim_t n, cumeric_t alpha, cumeric_t* x, dim_t incx);
}

#endif //YANN_LEVEL1_HPP
