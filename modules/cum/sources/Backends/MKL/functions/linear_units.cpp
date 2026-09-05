//
// Created by jaro on 7/19/26.
//
#include "cum/Core.hpp"
#include "cum/detail/vendor/oneapi/event_handler.hpp"
#include "internal/cumMKL.hpp"
#include "cum/memory.hpp"
#include "cum/functions/hyperbolic.hpp"
#include "cum/functions/linear_units.hpp"

#include <oneapi/mkl/vm/usm.hpp>

const cum::cumeric_t M_SQRT_2_OVER_PI = 0.79788456;
const cum::cumeric_t M_SQRT_2_OVER_PI_A = 0.044715;

namespace cum::functions::linear_units
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t identity(cumeric_t x)
    {
        return x;
    }

    cumeric_t relu(cumeric_t x)
    {
        return sycl::fmax(static_cast<cumeric_t>(x), static_cast<cumeric_t>(0));
    }

    cumeric_t leaky_relu(cumeric_t x)
    {
        return x > static_cast<cumeric_t>(0) ? x : leaky_relu_alpha * x;
    }

    cumeric_t gelu(cumeric_t x)
    {
        return static_cast<cumeric_t>(0.5) * x * (static_cast<cumeric_t>(1.0) + sycl::tanh(M_SQRT_2_OVER_PI * (x + M_SQRT_2_OVER_PI_A * x * x * x)));
    }

    cumeric_t elu(cumeric_t x)
    {
        return 0; // placeholder
    }

    /* Parallel versions */

    __event__ identity(cumeric_t* r, const cumeric_t* v, dim_t N)
    {
        // memory::memcopy(r, v, N);
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = v[idx];
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ relu(cumeric_t* r, const cumeric_t* v, const dim_t N)
    {
        sycl::event event = oneapi::mkl::vm::fmax(internal::getQueue(), N, v, internal::getZeros(), r, {});
        return detail::event_handler::create(std::move(event));
    }

    __event__ relu_in_place(cumeric_t* v, dim_t N)
    {
        sycl::event event = oneapi::mkl::vm::fmax(internal::getQueue(), N, v, internal::getZeros(), v, {});
        return detail::event_handler::create(std::move(event));
    }

    __event__ leaky_relu(cumeric_t* r, const cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = leaky_relu(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ leaky_relu_in_place(cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = leaky_relu(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ gelu(cumeric_t* r, const cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = gelu(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }
    __event__ gelu_in_place(cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = gelu(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ elu(cumeric_t* r, const cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = elu(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ elu_in_place(cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = elu(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    /* ========================== Derivatives ========================== */

    /* Scalar versions */

    cumeric_t identity_deriv(cumeric_t x)
    {
        return 1;
    }

    cumeric_t relu_deriv(cumeric_t x)
    {
        return x > static_cast<cumeric_t>(0) ? 1 : 0;
    }

    cumeric_t leaky_relu_deriv(cumeric_t x)
    {
        return x > static_cast<cumeric_t>(0) ? static_cast<cumeric_t>(1) : leaky_relu_alpha;
    }

    cumeric_t gelu_deriv(cumeric_t x)
    {
        cumeric_t tanh_arg = M_SQRT_2_OVER_PI * (x + M_SQRT_2_OVER_PI_A * x * x * x);
        cumeric_t tanh_val = sycl::tanh(tanh_arg);
        cumeric_t sech_squared = 1 - tanh_val * tanh_val;
        return static_cast<cumeric_t>(0.5) * (static_cast<cumeric_t>(1.0) + tanh_val) + (M_SQRT_2_OVER_PI * (x + 3 * M_SQRT_2_OVER_PI_A * x * x) * sech_squared) / static_cast<cumeric_t>(6.0);
    }

    cumeric_t elu_deriv(cumeric_t x)
    {
        return 0; // placeholder
    }

    /* Parallel versions */

    __event__ identity_deriv(cumeric_t* r, const cumeric_t* v, const dim_t N)
    {
        // memory::memcopy(r, internal::getOnes(), N);
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = static_cast<cumeric_t>(1);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ identity_deriv_in_place(cumeric_t* v, dim_t N)
    {
        // memory::memcopy(v, internal::getOnes(), N);
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = static_cast<cumeric_t>(1);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ relu_deriv(cumeric_t* r, const cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = relu_deriv(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ relu_deriv_in_place(cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = relu_deriv(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ leaky_relu_deriv(cumeric_t* r, const cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = leaky_relu_deriv(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ leaky_relu_deriv_in_place(cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = leaky_relu_deriv(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ gelu_deriv(cumeric_t* r, const cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = gelu_deriv(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ gelu_deriv_in_place(cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = gelu_deriv(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ elu_deriv(cumeric_t* r, const cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = gelu_deriv(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }

    __event__ elu_deriv_in_place(cumeric_t* v, dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = elu_deriv(v[idx]);
        });
        return detail::event_handler::create(std::move(event));
    }
}
