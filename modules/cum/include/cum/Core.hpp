#pragma once

#if defined(BUILD_USE_MKL)
    // #include <sycl/sycl.hpp>
#endif

#include <cstdint>

// #define __STDCPP_FLOAT16_T__
// #define __STDCPP_BFLOAT16_T__
#include <stdfloat>

namespace cum
{
    #if defined(BUILD_USE_MKL)
    // using cumeric_t = sycl::half;
    using cumeric_t = float;
    #else
    // using cumeric_t = _Float16;
    using cumeric_t = std::float_t;
    #endif
} // namespace cum
