//
// Created by jaro on 8/2/26.
//

#include <oneapi/mkl/blas.hpp>

#include <cstdint>

#include "cum/LinearAlgebra/BLAS/level1.hpp"
#include "cum/detail/vendor/oneapi/make_event.hpp"
#include "internal/cumMKL.hpp"

namespace cum::blas
{
    __event__ axpy(const dim_t n, const cumeric_t alpha, const cumeric_t* x, const dim_t incx, cumeric_t* y, const dim_t incy)
    {
        sycl::event event = oneapi::mkl::blas::row_major::axpy(internal::getQueue(), n, alpha, x, incx, y, incy);
        return detail::make_event::create(std::move(event));
    }

    __event__ axpy(const dim_t n, const cumeric_t alpha, const cumeric_t* x, const dim_t incx, const cumeric_t beta, cumeric_t* y, const dim_t incy)
    {
        sycl::event event = oneapi::mkl::blas::row_major::axpby(internal::getQueue(), n, alpha, x, incx, beta, y, incy);
        return detail::make_event::create(std::move(event));
    }

    __event__ dot(const dim_t n, const cumeric_t* x, const dim_t incx, const cumeric_t* y, const dim_t incy, cumeric_t* result)
    {
        sycl::event event = oneapi::mkl::blas::row_major::dot(internal::getQueue(), n, x, incx, y, incy, result);
        return detail::make_event::create(std::move(event));
    }

    __event__ asum(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        sycl::event event = oneapi::mkl::blas::row_major::asum(internal::getQueue(), n, x, incx, result);
        return detail::make_event::create(std::move(event));
    }

    __event__ nrm2(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        sycl::event event = oneapi::mkl::blas::row_major::nrm2(internal::getQueue(), n, x, incx, result);
        return detail::make_event::create(std::move(event));
    }

    __event__ iamax(const dim_t n, const cumeric_t* x, const dim_t incx, dim_t* result)
    {
        // std::int64_t tmp;
        sycl::event event = oneapi::mkl::blas::row_major::iamax(internal::getQueue(), n, x, incx, result);
        // result[0] = static_cast<dim_t>(tmp);
        return detail::make_event::create(std::move(event));
    }

    __event__ iamin(const dim_t n, const cumeric_t* x, const dim_t incx, dim_t* result)
    {
        // std::int64_t tmp;
        sycl::event event = oneapi::mkl::blas::row_major::iamin(internal::getQueue(), n, x, incx, result);
        // result[0] = static_cast<dim_t>(tmp);
        return detail::make_event::create(std::move(event));
    }

    __event__ amax(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        std::int64_t tmp;
        sycl::event event = oneapi::mkl::blas::row_major::iamax(internal::getQueue(), n, x, incx, &tmp);
        event.wait();
        result[0] = x[tmp];
        return detail::make_event::create(std::move(event));
    }

    __event__ amin(const dim_t n, const cumeric_t* x, const dim_t incx, cumeric_t* result)
    {
        std::int64_t tmp;
        sycl::event event = oneapi::mkl::blas::row_major::iamin(internal::getQueue(), n, x, incx, &tmp);
        event.wait();
        result[0] = x[tmp];
        return detail::make_event::create(std::move(event));
    }

	__event__ scal(const dim_t n, const cumeric_t alpha, cumeric_t* x, const dim_t incx)
	{
		sycl::event event = oneapi::mkl::blas::row_major::scal(internal::getQueue(),
			n, alpha, x, incx
		);
        return detail::make_event::create(std::move(event));
    }
}
