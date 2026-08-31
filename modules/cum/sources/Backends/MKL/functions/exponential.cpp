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

    void exp_in_place(cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::exp(internal::getQueue(), N, v, v, {});
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
        oneapi::mkl::vm::ln(internal::getQueue(), N, v, r, {});
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
        return sycl::log(static_cast<cumeric_t>(2)) * exp2(x);
    }

    cumeric_t exp2_deriv_from_result(cumeric_t x)
    {
        return sycl::log(static_cast<cumeric_t>(2)) * x;
    }

    cumeric_t exp10_deriv(cumeric_t x)
    {
        return sycl::log(static_cast<cumeric_t>(10)) * exp10(x);
    }

    cumeric_t exp10_deriv_from_result(cumeric_t x)
    {
        return sycl::log(static_cast<cumeric_t>(10)) * x;
    }

    cumeric_t log_deriv(cumeric_t x)
    {
        return static_cast<cumeric_t>(1) / x;
    }

    cumeric_t log2_deriv(cumeric_t x)
    {
        return static_cast<cumeric_t>(1) / (sycl::log(static_cast<cumeric_t>(2))*x);
    }

    cumeric_t log10_deriv(cumeric_t x)
    {
        return static_cast<cumeric_t>(1) / (sycl::log(static_cast<cumeric_t>(10))*x);
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
            r[idx] = sycl::log(static_cast<cumeric_t>(2)) * exp2(v[idx]);
        });
    }

    void exp2_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = sycl::log(static_cast<cumeric_t>(2)) * exp2(v[idx]);
        });
    }

    void exp10_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = sycl::log(static_cast<cumeric_t>(2)) * exp10(v[idx]);
        });
    }

    void exp10_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = sycl::log(static_cast<cumeric_t>(10)) * exp10(v[idx]);
        });
    }

    void log_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = log_deriv(v[idx]);
        });
    }

    void log_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = log_deriv(v[idx]);
        });
    }

    void log2_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = log2_deriv(v[idx]);
        });
    }

    void log2_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = log2_deriv(v[idx]);
        });
    }

    void log10_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = log10_deriv(v[idx]);
        });
    }

    void log10_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = log10_deriv(v[idx]);
        });
    }
}