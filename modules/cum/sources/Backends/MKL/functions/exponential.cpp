//
// Created by jaro on 7/19/26.
//

#include "cum/Core.hpp"
#include "internal/cumMKL.hpp"

#include <oneapi/mkl/vm.hpp>
#include <sycl/sycl.hpp>

namespace cum::functions::exponential
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t exp(cumeric_t x)
    {
        return sycl::exp(x);
    }

    cumeric_t exp2(cumeric_t x)
    {
        return sycl::exp2(x);
    }

    cumeric_t exp10(cumeric_t x)
    {
        return sycl::exp10(x);
    }

    cumeric_t log(cumeric_t x) // base e
    {
        return sycl::log(x);
    }

    cumeric_t log2(cumeric_t x)
    {
        return sycl::log2(x);
    }

    cumeric_t log10(cumeric_t x)
    {
        return sycl::log10(x);
    }

    /* Parallel versions */

    void exp(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::exp(internal::getQueue(), N, v, r, {});
    }

    void exp2(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::exp2(internal::getQueue(), N, v, r, {});
    }

    void exp10(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::exp2(internal::getQueue(), N, v, r, {});
    }

    void log(cumeric_t* r, const cumeric_t* v, const std::size_t N) // base e
    {
        oneapi::mkl::vm::log(internal::getQueue(), N, v, r, {});
    }

    void log2(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::log2(internal::getQueue(), N, v, r, {});
    }

    void log10(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::log10(internal::getQueue(), N, v, r, {});
    }

    /* ========================== Derivatives ========================== */

    /* Scalar versions */

    cumeric_t exp_deriv(cumeric_t x)
    {
        return sycl::exp(x);
    }

    cumeric_t exp_deriv_from_result(cumeric_t x)
    {
        return x;
    }

    cumeric_t exp2_deriv(cumeric_t x)
    {
        return sycl::log(2._c) * exp2(x);
    }

    cumeric_t exp2_deriv_from_result(cumeric_t x)
    {
        return sycl::log(2._c) * x;
    }

    cumeric_t exp10_deriv(cumeric_t x)
    {
        return sycl::log(10.0_c) * exp10(x);
    }

    cumeric_t exp10_deriv_from_result(cumeric_t x)
    {
        return sycl::log(10.0_c) * x;
    }

    cumeric_t log_derivative(cumeric_t x)
    {
        return 1._c / x;
    }

    cumeric_t log2_derivative(cumeric_t x)
    {
        return 1._c / (sycl::log(2._c)*x);
    }

    cumeric_t log10_derivative(cumeric_t x)
    {
        return 1._c / (sycl::log(10._c)*x);
    }

    /* Parallel versions */

    void exp_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        exp(r, v, N);
    }

    void exp_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        // literally do nothing
    }

    void exp2_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = sycl::log(2._c) * exp2(v[idx]);
        });
    }

    void exp2_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = sycl::log(2._c) * exp2(v[idx]);
        });
    }

    void exp10_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = sycl::log(2._c) * exp10(v[idx]);
        });
    }

    void exp10_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = sycl::log(10._c) * exp10(v[idx]);
        });
    }
}