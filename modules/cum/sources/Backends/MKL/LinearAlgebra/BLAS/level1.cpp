//
// Created by jaro on 8/2/26.
//

#include <oneapi/mkl/blas.hpp>

#include <cstdint>

#include "cum/LinearAlgebra/BLAS/level1.hpp"
#include "internal/cumMKL.hpp"

namespace cum::blas
{
    void axpy(const dim_t n, const cumeric_t alpha, const cumeric_t* x, const dim_t incx, cumeric_t* y, const dim_t incy)
    {
        oneapi::mkl::blas::row_major::axpy(internal::getQueue(), n, alpha, x, incx, y, incy);
    }

    void axpy(const dim_t n, const cumeric_t alpha, const cumeric_t* x, const dim_t incx, const cumeric_t beta, cumeric_t* y, const dim_t incy)
    {
        oneapi::mkl::blas::row_major::axpby(internal::getQueue(), n, alpha, x, incx, beta, y, incy);
    }

    void dot(const dim_t n, const cumeric_t* x, const dim_t incx, const cumeric_t* y, const dim_t incy, cumeric_t* result)
    {
        oneapi::mkl::blas::row_major::dot(internal::getQueue(), n, x, incx, y, incy, result);
    }

    void asum(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        oneapi::mkl::blas::row_major::asum(internal::getQueue(), n, x, incx, result);
    }

    void nrm2(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        oneapi::mkl::blas::row_major::nrm2(internal::getQueue(), n, x, incx, result);
    }

    void iamax(const dim_t n, const cumeric_t* x, const dim_t incx, dim_t* result)
    {
        std::int64_t tmp;
        oneapi::mkl::blas::row_major::iamax(internal::getQueue(), n, x, incx, &tmp).wait();
        result[0] = static_cast<dim_t>(tmp);
    }

    void iamin(const dim_t n, const cumeric_t* x, const dim_t incx, dim_t* result)
    {
        std::int64_t tmp;
        oneapi::mkl::blas::row_major::iamin(internal::getQueue(), n, x, incx, &tmp).wait();
        result[0] = static_cast<dim_t>(tmp);
    }

    void amax(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        std::int64_t tmp;
        oneapi::mkl::blas::row_major::iamax(internal::getQueue(), n, x, incx, &tmp).wait();
        result[0] = x[tmp];
    }

    void amin(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        std::int64_t tmp;
        oneapi::mkl::blas::row_major::iamin(internal::getQueue(), n, x, incx, &tmp).wait();
        result[0] = x[tmp];
    }

	void scal(const dim_t n, const cumeric_t alpha, cumeric_t* x, const dim_t incx)
	{
		oneapi::mkl::blas::row_major::scal(internal::getQueue(),
			n, alpha, x, incx
		).wait();
	}

}
