//
// Created by jaro on 7/19/26.
//

#include "cum/Core.hpp"
#include "internal/cumMKL.hpp"

#include <oneapi/mkl/vm.hpp>
#include <sycl/sycl.hpp>

namespace cum::functions::hyperbolic
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t tanh(cumeric_t x)
    {
        return sycl::tanh(x);
    }

    cumeric_t sinh(cumeric_t x)
    {
        return sycl::sinh(x);
    }

    cumeric_t cosh(cumeric_t x)
    {
        return sycl::cosh(x);
    }

    /* Parallel versions */

    void tanh(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::tanh(internal::getQueue(), N, v, r);
    }

    void tanh_in_place(cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::tanh(internal::getQueue(), N, v, v);
    }

    void sinh(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::sinh(internal::getQueue(), N, v, r);
    }

    void sinh_in_place(cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::sinh(internal::getQueue(), N, v, v);
    }

    void cosh(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::cosh(internal::getQueue(), N, v, r);
    }

    void cosh_in_place(cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::cosh(internal::getQueue(), N, v, v);
    }

    /* ========================== It's Derivatives ========================== */

    /* Scalar versions */

    cumeric_t tanh_deriv(cumeric_t x)
    {
        return static_cast<cumeric_t>(1) - hyperbolic::tanh(x) * hyperbolic::tanh(x);
    }

    cumeric_t tanh_deriv_from_result(cumeric_t x)
    {
        return static_cast<cumeric_t>(1) - x * x;
    }

    cumeric_t sinh_deriv(cumeric_t x)
    {
        return cosh(x);
    }

    cumeric_t cosh_deriv(cumeric_t x)
    {
        return sinh(x);
    }

    /* Parallel versions */

    void tanh_deriv(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        auto queue = internal::getQueue();
        auto e = oneapi::mkl::vm::tanh(queue, N, v, r);

        // queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        // {
        //     cum::cumeric_t tanh_squared = sycl::pow(sycl::tanh(v[idx]), 2);
        //     r[idx] = 1 - tanh_squared;
        //     // r[idx] = 1 - sycl::tanh(v[idx]) * sycl::tanh(v[idx]);
        // }).wait();
        queue.parallel_for(sycl::range<1>(N), {e},
        [=](sycl::id<1> idx)
        {
            r[idx] = 1 - r[idx] * r[idx];
        });
    }

    void tanh_deriv_from_result(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = 1 - v[idx] * v[idx];
        });
    }

    void tanh_deriv_from_result_in_place(cumeric_t* v, const std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = 1 - v[idx] * v[idx];
        });
    }

    void sinh_deriv(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::cosh(internal::getQueue(), N, v, r);
    }

    void sinh_deriv_in_place(cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::cosh(internal::getQueue(), N, v, v);
    }

    void cosh_deriv(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::sinh(internal::getQueue(), N, v, r);
    }

    void cosh_deriv_in_place(cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::sinh(internal::getQueue(), N, v, v);
    }

}