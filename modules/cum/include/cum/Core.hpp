#pragma once

#include <cstdint>
#include <stdfloat>

// #define CUM_USE_F16
// #define BUILD_USE_MKL

#if defined(BUILD_USE_MKL)
    #include <sycl/sycl.hpp>
#endif



namespace cum
{
    enum class CUM_DEVICE { AUTO, CPU, GPU };

    // give info ABOUT used precision in compile time for each precision
    #if defined(CUM_USE_F64)
    using cumeric_t = double;        
    #warning "F64"
    #elif defined(CUM_USE_F16)
        #if defined(BUILD_USE_MKL)
        using cumeric_t = sycl::half;
        #warning "F16 sycl format"
        #else
        using cumeric_t = _Float16;
        #endif
    #elif defined(CUM_USE_BF16)
        #if defined(BUILD_USE_MKL)
        using cumeric_t = sycl::bfloat16;
        #warning "BF16 sycl format"
        #else
        using cumeric_t = std::bfloat16;
        #warning "BF16"
        #endif
    #elif defined(CUM_USE_INT8)
    using cumeric_t = int8_t;
    #warning "INT8"
    #else
    using cumeric_t = float;
    #warning "F32"
    #endif

} // namespace cum


inline cum::cumeric_t operator"" _c(long double val) 
{
    return static_cast<cum::cumeric_t>(val);
}

inline cum::cumeric_t operator"" _c(unsigned long long val) 
{
    return static_cast<cum::cumeric_t>(val);
}