//
// Created by jaro on 8/2/26.
//

#ifndef YANN_LEVEL1_HPP
#define YANN_LEVEL1_HPP

#include "cum/Core.hpp"

namespace cum::blas
{
    /* vector-scalar ops */

    void axpy(dim_t n, cumeric_t alpha, const cumeric_t* x, dim_t incx, cumeric_t* y, dim_t incy);
    void axpy(dim_t n, cumeric_t alpha, const cumeric_t* x, dim_t incx, cumeric_t beta, cumeric_t* y, dim_t incy);

    /* vector-vector ops */

    void dot(dim_t n, const cumeric_t* x, dim_t incx, const cumeric_t* y, dim_t incy, cumeric_t* result);

    /* vector to scalar reductions */

    void asum(dim_t n, const cumeric_t* x, dim_t incx, cumeric_t* result);
    void nrm2(dim_t n, const cumeric_t* x, dim_t incx, cumeric_t* result);

    void iamax(dim_t n, const cumeric_t* x, dim_t incx, dim_t* result);
    void iamin(dim_t n, const cumeric_t* x, dim_t incx, dim_t* result);

    void amax(dim_t n, const cumeric_t* x, dim_t incx, cumeric_t* result);
    void amin(dim_t n, const cumeric_t* x, dim_t incx, cumeric_t* result);

	void scal(dim_t n, cumeric_t alpha, cumeric_t* x, dim_t incx);
}

#endif //YANN_LEVEL1_HPP
