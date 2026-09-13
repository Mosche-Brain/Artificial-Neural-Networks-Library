/*
 * @author: jaro
 * @name:   level3
 * @file:   modules/cum/bindings/C/include/cum/linear_algebra/blas/level3.h
 * @date:   13 September 2026 22:27:40
 */

#ifndef YANN_LEVEL3_H
#define YANN_LEVEL3_H

#include "cum/core.h"

__event__ cum_gemm(char transpose_a, char transpose_b, dim_t m, dim_t n, dim_t k, cumeric_t alpha, const cumeric_t* a, dim_t lda, const cumeric_t* b, dim_t ldb, cumeric_t beta, cumeric_t* c, dim_t ldc);

#endif //YANN_LEVEL3_H
