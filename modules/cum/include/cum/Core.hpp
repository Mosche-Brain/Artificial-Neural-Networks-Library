#pragma once

#include <cstdint>
#include <stdfloat>
#include <vector>


#if defined(BUILD_USE_MKL)
    #include <sycl/sycl.hpp>
    #include <sycl/ext/oneapi/bfloat16.hpp>
#endif

#if !defined(CUM_USE_F64)  && \
    !defined(CUM_USE_F32)  && \
    !defined(CUM_USE_F16)  && \
    !defined(CUM_USE_BF16) && \
    !defined(CUM_USE_INT8)
    #define CUM_USE_F32
#endif

#include "cum/datatypes.hpp"

#include "cum/experimental/__event__.hpp"

namespace cum
{
    enum class DEVICE { AUTO, CPU, GPU };

    using __event__ = experimental::__event__;
    using __depencies__ = std::vector<std::reference_wrapper<__event__>>;

    using dim_t = std::int64_t;
    using Shape = std::vector<cum::dim_t>;
    using Dims = std::vector<cum::dim_t>;

    enum class datatype : uint8_t { FP64, FP32, FP16, BF16, FP8, S64, S32, S16, S8, U64, U32, U16, U8, UNDEF };
    enum class layout : std::uint8_t {
        ANY,
        X,
        NC,
        OI,
        IO,
        NCHW,
        NHWC,
        OIHW,
        HWIO,
        NCDHW,
        NDHWC,
        TNC,
        NTC,
        STRIDED,
        UNDEF
    };

    // give info ABOUT used precision in compile time for each precision
    #if defined(CUM_USE_F64)
        using cumeric_t = double;        
        using cummulative_t = double;
        constexpr datatype default_type = datatype::FP64;
    #elif defined(CUM_USE_F32)
        using cumeric_t = float;
        constexpr cumeric_t EPSILON = 1e-9f;
        using cummulative_t = double;
        constexpr datatype default_type = datatype::FP32;
    #elif defined(CUM_USE_F16)
        #if defined(BUILD_USE_MKL)
        using cumeric_t = sycl::half;
        #else
        using cumeric_t = _Float16;
        #endif
        constexpr cumeric_t EPSILON = 1e-4f16;
        using cummulative_t = float;
        constexpr datatype default_type = datatype::FP16;
    #elif defined(CUM_USE_BF16)
        #if defined(BUILD_USE_MKL)
        using cumeric_t = sycl::ext::oneapi::bfloat16;
        #else
        using cumeric_t = std::bfloat16;
        constexpr cumeric_t EPSILON = 1e-3bf16;
        #endif
        using cummulative_t = float;
        constexpr datatype default_type = datatype::BF16;
    #elif defined(CUM_USE_INT8)
        using cumeric_t = int8_t;
        constexpr cumeric_t EPSILON = 1e1;
        using cummulative_t = float;
        constexpr datatype default_type = datatype::S8;
    #else
        #warning "Type was not defined"
    // #error "Data type didn't specified"
    #endif

    constexpr std::size_t datatype_size(datatype type)
    {
        switch (type)
        {
        case datatype::FP64: return 8;
        case datatype::FP32: return 4;
        case datatype::FP16: return 2;
        case datatype::BF16: return 2;
        case datatype::FP8:  return 1;

        case datatype::S64:  return 8;
        case datatype::S32:  return 4;
        case datatype::S16:  return 2;
        case datatype::S8:   return 1;

        case datatype::U64:  return 8;
        case datatype::U32:  return 4;
        case datatype::U16:  return 2;
        case datatype::U8:   return 1;

        case datatype::UNDEF:
        default:
            return 1;
        }
    }

    template <typename F>
    decltype(auto) dispatch_datatype(datatype dt, F&& f) // I must get better undestenting of these sematnics, this is realy strange
    {
        switch (dt)
        {
            case datatype::FP64: return f.template operator()<double>();
            case datatype::FP32: return f.template operator()<float>();
            #if defined(BUILD_USE_MKL)  
            case datatype::FP16: return f.template operator()<sycl::half>();
            case datatype::BF16: return f.template operator()<sycl::ext::oneapi::bfloat16>();
            #else
            #ifdef __STDCPP_FLOAT16_T__
            case datatype::FP16: return f.template operator()<std::float16_t>();
            #endif
            #ifdef __STDCPP_BFLOAT16_T__        
            case datatype::BF16: return f.template operator()<std::bfloat16_t>();
            #endif
            #endif
            case datatype::S64:  return f.template operator()<int64_t>();
            case datatype::S32:  return f.template operator()<int32_t>();
            case datatype::S16:  return f.template operator()<int16_t>();
            case datatype::S8:   return f.template operator()<int8_t>();
            case datatype::U64:  return f.template operator()<uint64_t>();
            case datatype::U32:  return f.template operator()<uint32_t>();
            case datatype::U16:  return f.template operator()<uint16_t>();
            case datatype::U8:   return f.template operator()<uint8_t>();
            default:
                throw std::runtime_error("Unsupported datatype in dispatch_datatype");
        }
    }
} // namespace cum


inline cum::cumeric_t operator"" _c(long double val) 
{
    return static_cast<cum::cumeric_t>(val);
}

inline cum::cumeric_t operator"" _c(unsigned long long val) 
{
    return static_cast<cum::cumeric_t>(val);
}