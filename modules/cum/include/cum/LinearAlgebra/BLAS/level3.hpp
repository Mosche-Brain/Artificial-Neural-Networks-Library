//
// Created by jaro on 8/2/26.
//

#ifndef YANN_LEVEL3_HPP
#define YANN_LEVEL3_HPP
#include "cum/Core.hpp"

#include "cum/LinearAlgebra/BLAS/types.hpp"

namespace cum::blas
{

    __event__ gemm(char transpose_a, char transpose_b, dim_t m, dim_t n, dim_t k, cumeric_t alpha, const cumeric_t* a, dim_t lda, const cumeric_t* b, dim_t ldb, cumeric_t beta, cumeric_t* c, dim_t ldc);
    __event__ symm(char horizontal, char vertical, dim_t m, dim_t n, dim_t k, cumeric_t alpha, const cumeric_t* a, dim_t lda, const cumeric_t* b, dim_t ldb, cumeric_t beta, cumeric_t* c, dim_t ldc);
    __event__ trmm(char horizontal, char vertical, char transpose_a, char diagonal, cumeric_t* c, dim_t m, dim_t n, cumeric_t alpha, const cumeric_t* a, dim_t lda, const cumeric_t* b, dim_t ldb);
}

#endif //YANN_LEVEL3_HPP
