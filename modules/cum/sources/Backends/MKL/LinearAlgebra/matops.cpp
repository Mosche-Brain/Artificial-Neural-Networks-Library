
#include <oneapi/mkl/blas.hpp>

#include "cum/Core.hpp"
#include "cum/runtime.hpp"
#include "internal/cumMKL.hpp"
#include "cum/LinearAlgebra/BLAS/types.hpp"

#include "cum/LinearAlgebra/matops.hpp"

namespace cum::LinearAlgebra
{
	void mmul(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, const dim_t m, const dim_t n, const dim_t k)
	{
		blas::gemm(
			static_cast<char>(blas::transpose::nontrans),
			static_cast<char>(blas::transpose::nontrans),
			m, n, k,
			1, A, k,
			B, n, 0,
			C, n
		);
	}

	void mmul(cumeric_t* A, const cumeric_t* B, const dim_t m, const dim_t n, const dim_t k)
	{
		mmul(A, A, B, m, n, k);
	}

	void mtrans(cumeric_t *At, const cumeric_t *A, const dim_t m, const dim_t n)
	{
		auto& q = internal::getQueue();
        #if defined(CUM_USE_F64) || defined(CUM_USE_F32)
        {
            oneapi::mkl::blas::row_major::omatcopy(q, oneapi::mkl::transpose::trans, m, n, 1.0, A, n, At, m);
        }
        #else // use custom transposition kernel
        {
            q.submit([=](sycl::handler& h){
                h.parallel_for(sycl::range<2>(m, n), [=](sycl::id<2> i){
                    At[i[1] * m + i[0]] = A[i[0] * n + i[1]];
                });
            });
        }
        #endif
	}

	void mtrans(cumeric_t* A, const dim_t m, const dim_t n)
	{
		mtrans(A, A, m, n);
	}
}
