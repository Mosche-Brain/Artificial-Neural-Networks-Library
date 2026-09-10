//
// Created by jaro on 7/19/26.
//

#include <oneapi/mkl/vm.hpp>
#include <sycl/sycl.hpp>

#include "cum/detail/vendor/oneapi/event_handler.hpp"

#include "internal/context.hpp"
#include "cum/functions/trigonometric.hpp"

namespace cum::functions::trigonometric
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t sin(cumeric_t x)
    {
        return sycl::sin(x);
    }


    cumeric_t cos(cumeric_t x)
    {
        return sycl::cos(x);
    }

    cumeric_t tan(cumeric_t x)
    {
        return sycl::tan(x);
    }

    /* Parallelized versions */

    __event__ sin(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        auto& q = internal::getQueue();
        sycl::event event = oneapi::mkl::vm::sin(q, N, v, r, {});
        return detail::event_handler::create(std::move(event));
        // q.wait();
    }

    __event__ sin_in_place(cumeric_t* v, const std::size_t N)
    {
        auto& q = internal::getQueue();
        sycl::event event = oneapi::mkl::vm::sin(q, N, v, v, {});
        // q.wait();

        // internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        // {
            // v[idx] = sycl::sin(v[idx]);
        // });
        return detail::event_handler::create(std::move(event));
    }

    __event__ cos(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        auto& q = internal::getQueue();
        sycl::event event = oneapi::mkl::vm::cos(q, N, v, r, {});
        return detail::event_handler::create(std::move(event));
        
        // q.wait();
    }

    __event__ cos_in_place(cumeric_t* v, const std::size_t N)
    {
        auto& q = internal::getQueue();
        sycl::event event = oneapi::mkl::vm::cos(q, N, v, v, {});
        return detail::event_handler::create(std::move(event));
        // q.wait();
    }

    __event__ tan(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::tan(cum::internal::getQueue(), N, v, r, {});
        return detail::event_handler::create(std::move(event));
    }

    __event__ tan_in_place(cumeric_t* v, std::size_t N)
    {
        sycl::event event = oneapi::mkl::vm::tan(cum::internal::getQueue(), N, v, v, {});
        return detail::event_handler::create(std::move(event));
    }

    /* ========================== Derivatives ========================== */

    /* Scalar versions */

    cumeric_t sin_deriv(cumeric_t x)
    {
        return sycl::cos(x);
    }

    cumeric_t cos_deriv(cumeric_t x)
    {
        return -sycl::sin(x);
    }

    cumeric_t tan_deriv(cumeric_t x)
    {
        cumeric_t t = sycl::tan(x);
        return 1._c + t * t;
    }

    cumeric_t tan_deriv_from_result(cumeric_t x)
    {
        return 1._c + x * x;
    }

    /* Parallelized versions */

    __event__ sin_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        cos(r, v, N);
    }

    __event__ sin_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        cos_in_place(v, N);
    }

    __event__ cos_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = - sycl::sin(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ cos_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = -sycl::sin(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ tan_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            cumeric_t t = tan(v[idx]);
            r[idx] = 1 + t * t;
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ tan_deriv_in_place(cumeric_t* v, std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            cumeric_t t = tan(v[idx]);
            v[idx] = 1 + t * t;
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ tan_deriv_from_result(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = 1 + v[idx] * v[idx];
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ tan_deriv_in_place_from_result(cumeric_t* v, std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = 1 + v[idx] * v[idx];
        });
        return detail::event_handler::create(std::move(event));
    }
}
