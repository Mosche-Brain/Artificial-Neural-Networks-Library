//
// Created by jaro on 8/2/26.
//

#ifndef YANN_LEVEL2_HPP
#define YANN_LEVEL2_HPP

namespace cum::blas
{
    void gemv(char transa, dim_t m, dim_t n, cumeric_t alpha, const cumeric_t* A, dim_t lda, const cumeric_t* x, dim_t incx, cumeric_t beta, cumeric_t* y, dim_t incy);

}

#endif //YANN_LEVEL2_HPP
