//
// Created by jaro on 8/2/26.
//

#include <oneapi/mkl/blas.hpp>
// #include <oneapi/mkl/blas/usm_decls.hpp>


#include "internal/cumMKL.hpp"
#include "cum/LinearAlgebra/BLAS/level3.hpp"

#include "../../internal/cumMKL.hpp"

namespace cum::blas
{
    void gemm(char transpose_a, char transpose_b, dim_t m, dim_t n, dim_t k, cumeric_t alpha, const cumeric_t* a, dim_t lda, const cumeric_t* b, dim_t ldb, cumeric_t beta, cumeric_t* c, dim_t ldc)
    {
        oneapi::mkl::blas::row_major::gemm(internal::getQueue(), static_cast<oneapi::mkl::transpose>(transpose_a), static_cast<oneapi::mkl::transpose>(transpose_b), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, oneapi::mkl::blas::compute_mode::standard, {});
    }

    void symm(char horizontal, char vertical, dim_t m, dim_t n, dim_t k, cumeric_t alpha, const cumeric_t* a, dim_t lda, const cumeric_t* b, dim_t ldb, cumeric_t beta, cumeric_t* c, dim_t ldc)
    {
        oneapi::mkl::blas::row_major::symm(internal::getQueue(), static_cast<oneapi::mkl::side>(horizontal), static_cast<oneapi::mkl::uplo>(vertical), m, n, alpha, a, lda, b, ldb, beta, c, ldc, oneapi::mkl::blas::compute_mode::standard, {});
    }

    void trmm(char horizontal, char vertical, char transpose_a, char diagonal, cumeric_t* c, dim_t m, dim_t n, cumeric_t alpha, const cumeric_t* a, dim_t lda, const cumeric_t* b, dim_t ldb)
    {
        // oneapi::mkl::blas::row_major::trmm(internal::getQueue(), static_cast<oneapi::mkl::side>(horizontal), static_cast<oneapi::mkl::uplo>(vertical), static_cast<oneapi::mkl::transpose>(transpose_a), static_cast<oneapi::mkl::diag>(diagonal), m, n, alpha, a, lda, b, ldb, oneapi::mkl::blas::compute_mode::standard, {});
    }
}
