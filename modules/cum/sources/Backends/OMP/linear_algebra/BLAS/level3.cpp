/*
 * @author: jaro
 * @name:   level3
 * @file:   modules/cum/sources/Backends/OMP/linear_algebra/BLAS/level3.cpp
 * @date:   12 September 2026 20:54:24
 */

#include <omp.h>

#include "cum/LinearAlgebra/BLAS/level1.hpp"

namespace cum::blas
{
    __event__ gemm(cumeric_t* r, const cumeric_t* a, const cumeric_t* b, const dim_t m, const dim_t n, const dim_t k)
    {
        for(dim_t i = 0; i < m; i++)
        {
            for(dim_t j = 0; j < n; j++)
            {
                cumeric_t sum = 0;
                #pragma omp for
                for(dim_t l = 0; l < k; l++)
                {
                    sum += a[i * k + l] * b[l * j + k];
                }
                r[i * n + j] = sum;
            }
        }
    }
}