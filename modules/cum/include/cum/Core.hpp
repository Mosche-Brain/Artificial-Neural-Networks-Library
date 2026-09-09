#pragma once

#include <cstdint>
#include <stdfloat>
#include <vector>


#if defined(BUILD_USE_MKL)
    #include <sycl/sycl.hpp>
#endif

#if !defined(CUM_USE_F64)  && \
    !defined(CUM_USE_F32)  && \
    !defined(CUM_USE_F16)  && \
    !defined(CUM_USE_BF16) && \
    !defined(CUM_USE_INT8)
    #define CUM_USE_F32
#endif

#include "cum/experimental/__event__.hpp"

namespace cum
{
    enum class CUM_DEVICE { AUTO, CPU, GPU };

    using __event__ = experimental::__event__;
    using __depencies__ = std::vector<std::reference_wrapper<__event__>>;
    // using 
    // using __event__ = void;

    // give info ABOUT used precision in compile time for each precision
    #if defined(CUM_USE_F64)
        using cumeric_t = double;        
        // #warning "F64"
    #elif defined(CUM_USE_F32)
        using cumeric_t = float;
        constexpr cumeric_t EPSILON = 1e-9f;
        // #warning "F32"
    #elif defined(CUM_USE_F16)
        #if defined(BUILD_USE_MKL)
        using cumeric_t = sycl::half;
        // #warning "F16 sycl format"
        #else
        using cumeric_t = _Float16;
        #endif
        constexpr cumeric_t EPSILON = 1e-4f16;
    #elif defined(CUM_USE_BF16)
        #if defined(BUILD_USE_MKL)
        using cumeric_t = sycl::bfloat16;
        // #warning "BF16 sycl format"
        #else
        using cumeric_t = std::bfloat16;
        constexpr cumeric_t EPSILON = 1e-3bf16;
        #warning "BF16"
        #endif
    #elif defined(CUM_USE_INT8)
        using cumeric_t = int8_t;
        constexpr cumeric_t EPSILON = 1e1;
        // #warning "INT8"
    #else
        #warning "Type was not defined"
    // #error "Data type didn't specified"
    #endif

    using dim_t = std::int64_t;

    #if defined(CUM_USE_F32)
    using cummulative_t = float;
    #else
    using cummulative_t = double;
    #endif

    using Shape = std::vector<cum::dim_t>;

} // namespace cum


inline cum::cumeric_t operator"" _c(long double val) 
{
    return static_cast<cum::cumeric_t>(val);
}

inline cum::cumeric_t operator"" _c(unsigned long long val) 
{
    return static_cast<cum::cumeric_t>(val);
}