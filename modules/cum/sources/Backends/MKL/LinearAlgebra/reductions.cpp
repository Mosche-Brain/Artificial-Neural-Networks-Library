//
// Created by jaro on 8/2/26.
//

#include "internal/cumMKL.hpp"
#include "cum/LinearAlgebra/BLAS/level1.hpp"
#include "cum/LinearAlgebra/BLAS/level2.hpp"
#include "cum/LinearAlgebra/BLAS/types.hpp"

#include "cum/LinearAlgebra/reductions.hpp"

namespace cum::LinearAlgebra
{
    /* Matrix ---> Vector */

    void rwise_sum(cumeric_t* r, cumeric_t* mat, dim_t m, dim_t n)
    {
        #if defined(CUM_USE_F64) || defined(CUM_USE_F32)
        {
            blas::gemv(
                static_cast<char>(blas::transpose::nontrans),
                m, n, 1.0,
                mat, n, internal::getOnes(),
                1, 0.0, r, 1
            );
        }
        #else
        #endif
    }

    void cwise_sum(cumeric_t* r, cumeric_t* mat, dim_t m, dim_t n)
    {

    }

    /* Vector ---> Scalar */

    void asum(cumeric_t* r, const cumeric_t* v, dim_t n)
    {
        blas::asum(n, v, 1, r);
    }

    void sum(cumeric_t* r, const cumeric_t* v, const dim_t n)
    {
        auto reduction = sycl::reduction(
            r, sycl::plus<cumeric_t>()
        );

        internal::getQueue().parallel_for(sycl::range<1>(n), reduction, [=](sycl::id<1> idx, auto& sum)
        {
            sum.combine(v[idx]);
        });
    }

    void mean(cumeric_t* r, const cumeric_t* v, dim_t n)
    {
        sum(r, v, n);
		r[0] /= static_cast<cumeric_t>(n);
    }

    void amean(cumeric_t* r, const cumeric_t* v, dim_t n)
    {
		asum(r, v, n);
		r[0] /= static_cast<cumeric_t>(n);
    }
}
