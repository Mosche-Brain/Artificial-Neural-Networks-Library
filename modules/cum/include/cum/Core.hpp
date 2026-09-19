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

#include "cum/experimental/__event__.hpp"
#include "cum/layout.hpp"
#include "cum/datatypes.hpp"


namespace cum
{
    enum class DEVICE { AUTO, CPU, GPU };

    using __event__ = experimental::__event__;
    using __events__ = std::vector<std::reference_wrapper<__event__>>;

    using dim_t = std::int64_t;
    using Shape = std::vector<cum::dim_t>;
    using Dims = std::vector<cum::dim_t>;


    // give info ABOUT used precision in compile time for each precision

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