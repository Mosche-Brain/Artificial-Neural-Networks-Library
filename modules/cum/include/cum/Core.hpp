#pragma once

#include "cum/config.hpp"
#include "cum/datatypes.hpp"
#include <cstdint>
#include <stdfloat>
#include <vector>

#if CUM_USE_MKL
    #include <sycl/sycl.hpp>
    #include <sycl/ext/oneapi/bfloat16.hpp>
#endif

#if !CUM_USE_FP64 && \
    !CUM_USE_FP32 && \
    !CUM_USE_FP16 && \
    !CUM_USE_BF16 && \
    !CUM_USE_INT8
    #error "No CUM numeric type was configured"
#endif

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
    #if CUM_USE_FP64
        using cumeric_t = double;
        using cummulative_t = double;
        constexpr datatype default_type = datatype::FP64;
    #elif CUM_USE_FP32
        using cumeric_t = float;
        constexpr cumeric_t EPSILON = 1e-9f;
        using cummulative_t = double;
        constexpr datatype default_type = datatype::FP32;
    #elif CUM_USE_FP16
        using cumeric_t = cum::float16;
        constexpr cumeric_t EPSILON = 1e-4f16;
        using cummulative_t = float;
        constexpr datatype default_type = datatype::FP16;
    #elif CUM_USE_BF16
        using cumeric_t = cum::bfloat16;
        constexpr cumeric_t EPSILON = 1e-3bf16;
        using cummulative_t = float;
        constexpr datatype default_type = datatype::BF16;
    #elif CUM_USE_INT8
        using cumeric_t = int8_t;
        constexpr cumeric_t EPSILON = 1e1;
        using cummulative_t = float;
        constexpr datatype default_type = datatype::S8;
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
    decltype(auto) dispatch_datatype(datatype dt, F&& f)
    {
        switch (dt)
        {
            case datatype::FP64: return f.template operator()<double>();
            case datatype::FP32: return f.template operator()<float>();
            case datatype::FP16: return f.template operator()<cum::float16>();
            case datatype::BF16: return f.template operator()<cum::bfloat16>();
            case datatype::FP8:  return f.template operator()<cum::float8>();
            case datatype::S64:  return f.template operator()<std::int64_t>();
            case datatype::S32:  return f.template operator()<std::int32_t>();
            case datatype::S16:  return f.template operator()<std::int16_t>();
            case datatype::S8:   return f.template operator()<std::int8_t>();
            case datatype::U64:  return f.template operator()<std::uint64_t>();
            case datatype::U32:  return f.template operator()<std::uint32_t>();
            case datatype::U16:  return f.template operator()<std::uint16_t>();
            case datatype::U8:   return f.template operator()<std::uint8_t>();
            default:             return f.template operator()<cum::cumeric_t>();
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