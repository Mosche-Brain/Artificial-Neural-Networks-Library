//
// Created by jaro on 7/19/26.
//

#include "cum/Core.hpp"
#include "cum/detail/vendor/oneapi/event_handler.hpp"
#include "internal/context.hpp"

#include <oneapi/mkl/vm.hpp>
#include <sycl/sycl.hpp>
#include <utility>

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

    __event__ tanh(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::tanh(internal::getQueue(), N, v, r);
        return detail::event_handler::create(std::move(event));
    }

    __event__ tanh_in_place(cumeric_t* v, const std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::tanh(internal::getQueue(), N, v, v);
        return detail::event_handler::create(std::move(event));
    }

    __event__ sinh(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::sinh(internal::getQueue(), N, v, r);
        return detail::event_handler::create(std::move(event));
    }

    __event__ sinh_in_place(cumeric_t* v, const std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::sinh(internal::getQueue(), N, v, v);
        return detail::event_handler::create(std::move(event));
        return detail::event_handler::create(std::move(event));
    }

    __event__ cosh(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::cosh(internal::getQueue(), N, v, r);
        return detail::event_handler::create(std::move(event));
    }

    __event__ cosh_in_place(cumeric_t* v, const std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::cosh(internal::getQueue(), N, v, v);
        return detail::event_handler::create(std::move(event));
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

    __event__ tanh_deriv(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        auto queue = internal::getQueue();
        sycl::event e = oneapi::mkl::vm::tanh(queue, N, v, r);

        // queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        // {
        //     cum::cumeric_t tanh_squared = sycl::pow(sycl::tanh(v[idx]), 2);
        //     r[idx] = 1 - tanh_squared;
        //     // r[idx] = 1 - sycl::tanh(v[idx]) * sycl::tanh(v[idx]);
        // }).wait();
        sycl::event event = queue.parallel_for(sycl::range<1>(N), {e},
        [=](sycl::id<1> idx)
        {
            r[idx] = 1 - r[idx] * r[idx];
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ tanh_deriv_from_result(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = 1 - v[idx] * v[idx];
        });
    }

    __event__ tanh_deriv_from_result_in_place(cumeric_t* v, const std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = 1 - v[idx] * v[idx];
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ sinh_deriv(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::cosh(internal::getQueue(), N, v, r);
        return detail::event_handler::create(std::move(event));
    }

    __event__ sinh_deriv_in_place(cumeric_t* v, const std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::cosh(internal::getQueue(), N, v, v);
        return detail::event_handler::create(std::move(event));
    }

    __event__ cosh_deriv(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::sinh(internal::getQueue(), N, v, r);
        return detail::event_handler::create(std::move(event));
    }

    __event__ cosh_deriv_in_place(cumeric_t* v, const std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::sinh(internal::getQueue(), N, v, v);
        return detail::event_handler::create(std::move(event));
    }

}