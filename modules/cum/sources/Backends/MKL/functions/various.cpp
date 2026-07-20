//
// Created by jaro on 7/19/26.
//

#include <sycl/sycl.hpp>

#include "internal/cumMKL.hpp"
#include "cum/functions/various.hpp"

namespace cum::functions::various
{
    void fill(cumeric_t* v, cumeric_t val, const std::size_t N)
    {
        internal::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            v[i] = val;
        }).wait();
    }

    cumeric_t min(const cumeric_t a, const cumeric_t b) { return sycl::min(a, b); }

    cumeric_t max(const cumeric_t a, const cumeric_t b) { return sycl::max(a, b); }

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

}