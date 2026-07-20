/* Created by jaro on 7/19/26. */

#include <oneapi/mkl/vm.hpp>
#include <sycl/sycl.hpp>

#include "internal/cumMKL.hpp"
#include "cum/functions/cyclometric.hpp"

namespace cum::functions::cyclometric
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t asin(cumeric_t x)
    {
        return sycl::asin(x);
    }

    cumeric_t acos(cumeric_t x)
    {
        return sycl::acos(x);
    }

    cumeric_t atan(cumeric_t x)
    {
        return sycl::atan(x);
    }

    /* Parallel versions */

    void asin(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        oneapi::mkl::vm::asin(internal::getQueue(), N, v, r, {});
    }

    void asin_in_place(cumeric_t* v, std::size_t N)
    {
        oneapi::mkl::vm::asin(internal::getQueue(), N, v, v, {});
    }

    void acos(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        oneapi::mkl::vm::acos(internal::getQueue(), N, v, r, {});
    }

    void acos_in_place(cumeric_t* v, std::size_t N)
    {
        oneapi::mkl::vm::acos(internal::getQueue(), N, v, v, {});
    }

    void atan(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        oneapi::mkl::vm::atan(internal::getQueue(), N, v, r, {});
    }

    void atan_in_place(cumeric_t* v, std::size_t N)
    {
        oneapi::mkl::vm::atan(internal::getQueue(), N, v, v, {});
    }

    /* ========================== Derivatives ========================== */

    /* Scalar versions */

    cumeric_t asin_deriv(cumeric_t x)
    {
        return static_cast<cumeric_t>(1) / sycl::sqrt(static_cast<cumeric_t>(1) - x*x);
    }

    cumeric_t acos_deriv(cumeric_t x)
    {
        return static_cast<cumeric_t>(-1) / sycl::sqrt(static_cast<cumeric_t>(1) - x*x);
    }
    cumeric_t atan_deriv(cumeric_t x)
    {
        return static_cast<cumeric_t>(1)/ (x*x + 1);
    }

    /* Parallel versions */

    void asin_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = asin_deriv(v[idx]);
        });
    }

    void asin_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = asin_deriv(v[idx]);
        });
    }

    void acos_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = acos_deriv(v[idx]);
        });
    }

    void acos_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = acos_deriv(v[idx]);
        });
    }

    void atan_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = atan_deriv(v[idx]);
        });
    }

    void atan_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = atan_deriv(v[idx]);
        });
    }
}