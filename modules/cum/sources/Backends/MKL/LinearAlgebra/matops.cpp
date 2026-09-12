
#include <oneapi/mkl/blas.hpp>
#include <vector>

#include "cum/Core.hpp"
#include "cum/runtime.hpp"

#include "cum/detail/vendor/oneapi/event_handler.hpp"
#include "internal/context.hpp"
#include "cum/LinearAlgebra/BLAS/types.hpp"

#include "cum/LinearAlgebra/matops.hpp"

namespace cum::LinearAlgebra
{
	__event__ mmul(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, const dim_t m, const dim_t n, const dim_t k, __depencies__ depencies)
	{
		// std::vector<sycl::event> events;
		// for(__event__& dependency : depencies)
		// 	events.push_back(dependency.);

		return blas::gemm
		(
			static_cast<char>(blas::transpose::nontrans),
			static_cast<char>(blas::transpose::nontrans),
			m, n, k,
			1, A, k, 
			B, n, 0,
			C, n
		);
	}

	__event__ mmul(cumeric_t* A, const cumeric_t* B, const dim_t m, const dim_t n, const dim_t k)
	{
		return mmul(A, A, B, m, n, k);
	}

	__event__ mtrans(cumeric_t *At, const cumeric_t *A, const dim_t m, const dim_t n, __depencies__ depencies)
	{
		auto& q = internal::getQueue();
        sycl::event event;
		#if CUM_USE_FP64 || CUM_USE_FP32
        {
            event = oneapi::mkl::blas::row_major::omatcopy(q, oneapi::mkl::transpose::trans, m, n, 1.0, A, n, At, m);
        }
        #else // use custom transposition kernel
        {
            event = q.submit([=](sycl::handler& h)
			{
                h.parallel_for
				(
					sycl::range<2>(m, n), [=](sycl::id<2> i)
					{
                    	At[i[1] * m + i[0]] = A[i[0] * n + i[1]];
                	}
				);
            });
        }
        #endif
		return detail::event_handler::create(std::move(event));
	}

	__event__ mtrans(cumeric_t* A, const dim_t m, const dim_t n, __depencies__ depencies)
	{
		return mtrans(A, A, m, n, depencies);
	}
}
