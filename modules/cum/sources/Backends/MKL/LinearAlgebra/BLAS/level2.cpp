//
// Created by jaro on 8/2/26.
//

#include <oneapi/mkl/blas.hpp>

#include "internal/context.hpp"

#include "cum/LinearAlgebra/BLAS/level2.hpp"

#include "cum/LinearAlgebra/BLAS/types.hpp"

namespace cum::blas
{
    __event__ gemv(const char transa, const dim_t m, const dim_t n, const cumeric_t alpha, const cumeric_t* A, const dim_t lda, const cumeric_t* x, dim_t incx, const cumeric_t beta, cumeric_t* y, const dim_t incy)
    {
        oneapi::mkl::blas::row_major::gemv(internal::getQueue(),
            static_cast<oneapi::mkl::transpose>(transa),
            m, n,alpha,
            A, lda, x,
            incx, beta, y,
            incy
        );
    }

    __event__ symv(const char vertical, const dim_t n, const cumeric_t alpha, const cumeric_t* A, const dim_t lda, const cumeric_t* x, const dim_t incx, const cumeric_t beta, cumeric_t* y, const dim_t incy)
    {

    }

    __event__ trmv(const char vertical, const char transa, const char diag, dim_t n, const cumeric_t* A, dim_t lda, cumeric_t* x, dim_t incx)
    {

    }
}
