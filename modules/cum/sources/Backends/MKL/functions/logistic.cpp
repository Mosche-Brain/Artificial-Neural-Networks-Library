/* Created by jaro on 7/19/26. */

#include "cum/Core.hpp"

#include "cum/LinearAlgebra/BLAS/level1.hpp"
#include "cum/LinearAlgebra/reductions.hpp"
#include "cum/functions/exponential.hpp"

#include "internal/cumMKL.hpp"
#include "cum/detail/vendor/oneapi/make_event.hpp"


#include "cum/functions/logistic.hpp"

namespace cum::functions::logistic
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t sigmoid(cumeric_t x)
    {
        return static_cast<cumeric_t>(1.0) / (static_cast<cumeric_t>(1.0) + sycl::exp(-x));
    }

    /* Parallel versions */

    __event__ sigmoid(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = sigmoid(v[idx]);
        });
        return detail::make_event::create(std::move(event));
    }

    __event__ sigmoid_in_place(cumeric_t* v, const std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = sigmoid(v[idx]);
        });
        return detail::make_event::create(std::move(event));
    }

    __event__ softmax(cumeric_t* r, const cumeric_t* v, const dim_t N)
    {
        exponential::exp(r, v, N);
        
        cumeric_t sum_exp_v = 0;
        LinearAlgebra::sum(&sum_exp_v, r, N);

        return blas::scal(N, 1/sum_exp_v, r, 1);
        // sycl::event event = blas::scal(N, 1/sum_exp_v, r, 1);
        // return detail::make_event::create(std::move(event));
    }

    __event__ softmax_in_place(cumeric_t* v, const dim_t N)
    {
        // sycl::event event = exponential::exp_in_place(v, N);
        return exponential::exp_in_place(v, N);

        // return detail::make_event::create(std::move(event));
    }

    /* ========================== Derivatives ========================== */

    /* Scalar versions */

    cumeric_t sigmoid_deriv(cumeric_t x)
    {
        return sigmoid(x) * (static_cast<cumeric_t>(1.0) - sigmoid(x));
    }

    cumeric_t sigmoid_deriv_from_result(cumeric_t x)
    {
        return x * (static_cast<cumeric_t>(1.0) - x);
    }

    /* Parallel versions */

    __event__ sigmoid_deriv(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = sigmoid_deriv(v[idx]);
        });
        return detail::make_event::create(std::move(event));
    }

    __event__ sigmoid_deriv_from_result(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = sigmoid_deriv_from_result(v[idx]);
        });
        return detail::make_event::create(std::move(event));
    }

    __event__ sigmoid_deriv_in_place(cumeric_t* v, const std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = sigmoid_deriv(v[idx]);
        });
        return detail::make_event::create(std::move(event));
    }

    __event__ sigmoid_deriv_from_result_in_place(cumeric_t* v, const std::size_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = sigmoid_deriv(v[idx]);
        });
        return detail::make_event::create(std::move(event));
    }

    __event__ softmax_deriv(cumeric_t* r, const cumeric_t* v, const dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            r[idx] = sigmoid_deriv(v[idx]);
        });
        return detail::make_event::create(std::move(event));
    }

    __event__ softmax_deriv_in_place(cumeric_t* v, const dim_t N)
    {
        sycl::event event = internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            v[idx] = sigmoid_deriv(v[idx]);
        });
        return detail::make_event::create(std::move(event));
    }
    
    __event__ softmax_deriv_from_result(cumeric_t* r, const cumeric_t* v, const dim_t N)
    {

    }

}
