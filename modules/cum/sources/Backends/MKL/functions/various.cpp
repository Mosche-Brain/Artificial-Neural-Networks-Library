//
// Created by jaro on 7/19/26.
//

#include <sycl/sycl.hpp>

#include "internal/context.hpp"
#include "cum/functions/various.hpp"

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include "cum/memory.hpp"

namespace cum::functions::various
{
    void fill(cumeric_t* v, const cumeric_t val, const std::size_t N)
    {
        // internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        // {
            // const std::size_t i = idx[0];
            // v[idx] = val;
        // }).wait();

        if (N == 0) return;

        sycl::ext::oneapi::experimental::fill(internal::getQueue(), v, val, N);
    }

    cumeric_t clamp(cumeric_t a, cumeric_t min, cumeric_t max)
    {
        return sycl::clamp(a, min, max);
    }

    // cummulative_t clamp(cummulative_t a, cummulative_t min, cummulative_t max)
    // {
    //     return sycl::clamp(a, min, max);
    // }

    cumeric_t min(const cumeric_t a, const cumeric_t b) { return sycl::min(a, b); }

    cumeric_t max(const cumeric_t a, const cumeric_t b) { return sycl::max(a, b); }

    cumeric_t abs(cumeric_t x)
    {
        return sycl::fabs(x);
    }

    // temporary naive implementation
    void clip(cumeric_t* r, const cumeric_t* v, const cumeric_t min, const cumeric_t max, const std::size_t N)
    {
        for(size_t i = 0 ; i < N ; i++) { r[i] = v[i] > min && v[i] < max ? v[i] : v[i] < min ? min : max; };
    }

    // temporary naive implementation
    void clipInPlace(cumeric_t* v, const cumeric_t min, const cumeric_t max, const std::size_t N)
    {
        for(size_t i = 0 ; i < N ; i++) { v[i] = v[i] > min && v[i] < max ? v[i] : v[i] < min ? min : max; };
    }


    bool equal(const cumeric_t* v, const  cumeric_t* u, const std::size_t N)
    {
        auto policy = oneapi::dpl::execution::make_device_policy(internal::getQueue());
        return std::equal(policy, v, v + N, u);
    }

    void linespace(cumeric_t* v, const cumeric_t start, const cumeric_t end, const std::size_t N)
    {
        // cumeric_t* buff = cum::memory::allocate(N);
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];

            if (N == 1)
            {
                v[i] = start;
            }
            else
            {
                v[i] = start + static_cast<cumeric_t>(i) * (end - start) / static_cast<cumeric_t>(N - 1);
            }
        }).wait();
    }
}
