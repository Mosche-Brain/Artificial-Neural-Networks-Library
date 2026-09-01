//
// Created by jaro on 8/30/26.
//

#include <oneapi/mkl/blas.hpp>

#include "internal/cumMKL.hpp"

#include "cum/LinearAlgebra/vecops.hpp"


namespace cum::LinearAlgebra
{
	__event__ vadd(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		// auto copy = q.copy(v, r, N);
		// oneapi::mkl::blas::row_major::axpy(q, N, 1.0, u, 1, r, 1, {});
		oneapi::mkl::vm::add(internal::getQueue(),
				n, v, u, r
		).wait();
	}

	__event__ vadd(cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::add(internal::getQueue(),
			n, v, u, v
		).wait();
	}

	__event__ vsub(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::sub(internal::getQueue(),
			n, v, u, r
		).wait();
	}

	__event__ vsub(cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::sub(internal::getQueue(),
			n, v, u, v
		).wait();
	}

	__event__ vmul(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::mul(internal::getQueue(),
			n, v, u, r
		).wait();
	}

	__event__ vmul(cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::mul(internal::getQueue(),
			n, v, u, v
		).wait();
	}

	__event__ vdiv(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::div(internal::getQueue(),
			n, v, u, r
		).wait();	}

	__event__ vdiv(cumeric_t* v, const cumeric_t* u, const dim_t n)
	{
		oneapi::mkl::vm::div(internal::getQueue(),
			n, v, u, v
		).wait();
	}
}
