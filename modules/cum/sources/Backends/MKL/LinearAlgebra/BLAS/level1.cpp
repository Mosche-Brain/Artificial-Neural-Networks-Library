//
// Created by jaro on 8/2/26.
//

#include <oneapi/mkl/blas/usm.hpp>
#include <oneapi/mkl/blas/usm_decls.hpp>


#include "internal/cumMKL.hpp"
#include "cum/LinearAlgebra/BLAS/level3.hpp"

#include "../../internal/cumMKL.hpp"


#include "cum/LinearAlgebra/BLAS/level1.hpp"

namespace cum::blas
{
    void axpy(const dim_t n, const cumeric_t alpha, const cumeric_t* x, const dim_t incx, cumeric_t* y, const dim_t incy)
    {
        oneapi::mkl::blas::row_major::axpy(internal::getQueue(), n, alpha, x, incx, y, incy);
    }

    void axpby(const dim_t n, const cumeric_t alpha, const cumeric_t* x, const dim_t incx, const cumeric_t beta, cumeric_t* y, const dim_t incy)
    {
        oneapi::mkl::blas::row_major::axpby(internal::getQueue(), n, alpha, x, incx, beta, y, incy);
    }

    void dot(dim_t n, cumeric_t* x, const dim_t incx, const cumeric_t* y, const dim_t incy, cumeric_t* result)
    {
        oneapi::mkl::blas::row_major::dot(internal::getQueue(), n, x, incx, y, incy, result);
    }

    void asum(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        oneapi::mkl::blas::row_major::asum(internal::getQueue(), n, x, incx, result);
    }

    void nrm2(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        oneapi::mkl::blas::row_major::nrm2(internal::getQueue(), n, x, incx, result);
    }

    void iamax(const dim_t n, const cumeric_t* x, const dim_t incx, dim_t* result)
    {
        oneapi::mkl::blas::row_major::iamax(internal::getQueue(), n, x, incx, result);
    }

    void iamin(const dim_t n, const cumeric_t* x, const dim_t incx, dim_t* result)
    {
        oneapi::mkl::blas::row_major::iamin(internal::getQueue(), n, x, incx, result);
    }

    void amax(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        dim_t tmp;
        oneapi::mkl::blas::row_major::iamax(internal::getQueue(), n, x, incx, &tmp);
        result[0] = x[tmp];
    }

    void amin(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        dim_t tmp;
        oneapi::mkl::blas::row_major::iamin(internal::getQueue(), n, x, incx, &tmp);
        result[0] = x[tmp];
    }

}