//
// Created by jaro on 8/30/26.
//

#include <oneapi/mkl/blas.hpp>

#include "internal/cumMKL.hpp"

#include "cum/LinearAlgebra/vecops.hpp"


namespace cum::LinearAlgebra
{
	void vadd(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		// auto copy = q.copy(v, r, N);
		// oneapi::mkl::blas::row_major::axpy(q, N, 1.0, u, 1, r, 1, {});
		oneapi::mkl::vm::add(internal::getQueue(),
				n, v, u, r
		).wait();
	}

	void vadd(cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::add(internal::getQueue(),
			n, v, u, v
		).wait();
	}

	void vsub(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::sub(internal::getQueue(),
			n, v, u, r
		).wait();
	}

	void vsub(cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::sub(internal::getQueue(),
			n, v, u, v
		).wait();
	}

	void vmul(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::mul(internal::getQueue(),
			n, v, u, r
		).wait();
	}

	void vmul(cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::mul(internal::getQueue(),
			n, v, u, v
		).wait();
	}

	void vdiv(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::div(internal::getQueue(),
			n, v, u, r
		).wait();	}

	void vdiv(cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::div(internal::getQueue(),
			n, v, u, v
		).wait();
	}
}
