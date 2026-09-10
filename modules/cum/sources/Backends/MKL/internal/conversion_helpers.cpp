//
// Created by jaro on 9/10/26.
//

#include "cum/detail/vendor/oneapi/conversion_helpers.hpp"

namespace cum
{
    dnnl::memory::data_type dnnl_data_type(datatype dtype)
    {
        switch(dtype)
        {
            case datatype::FP64: return dnnl::memory::data_type::f64;
            case datatype::FP32: return dnnl::memory::data_type::f32;
            case datatype::FP16: return dnnl::memory::data_type::f16;
            case datatype::BF16: return dnnl::memory::data_type::bf16;
            case datatype::FP8:  return dnnl::memory::data_type::f8_e4m3;
            case datatype::S64:   return dnnl::memory::data_type::s64;
            case datatype::S32:   return dnnl::memory::data_type::s32;
            case datatype::S16:   return dnnl::memory::data_type::undef;
            case datatype::S8:   return dnnl::memory::data_type::s8;
            case datatype::U64:   return dnnl::memory::data_type::undef;
            case datatype::U32:   return dnnl::memory::data_type::undef;
            case datatype::U16:   return dnnl::memory::data_type::undef;
            case datatype::U8:   return dnnl::memory::data_type::u8;
            default: return dnnl::memory::data_type::undef;
        }
    }

    datatype cum_data_type(dnnl::memory::data_type dtype)
    {
        switch(dtype)
        {
            case dnnl::memory::data_type::f64:      return datatype::FP64;
            case dnnl::memory::data_type::f32:      return datatype::FP32;
            case dnnl::memory::data_type::f16:      return datatype::FP16;
            case dnnl::memory::data_type::bf16:     return datatype::BF16;
            case dnnl::memory::data_type::f8_e4m3:  return datatype::FP8;
            case dnnl::memory::data_type::s8:       return datatype::S8;
            case dnnl::memory::data_type::u8:       return datatype::U8;
            case dnnl::memory::data_type::s32:  return datatype::S32;
            default:                            return datatype::UNDEF;
        }
    }

    dnnl::memory::format_tag dnnl_format_tag(layout format)
    {
        switch(format)
        {
            case layout::ANY:     return dnnl::memory::format_tag::any;
            case layout::X:       return dnnl::memory::format_tag::x;
            case layout::NC:      return dnnl::memory::format_tag::nc;
            case layout::OI:      return dnnl::memory::format_tag::oi;
            case layout::IO:      return dnnl::memory::format_tag::io;
            case layout::NCHW:    return dnnl::memory::format_tag::nchw;
            case layout::NHWC:    return dnnl::memory::format_tag::nhwc;
            case layout::OIHW:    return dnnl::memory::format_tag::oihw;
            case layout::HWIO:    return dnnl::memory::format_tag::hwio;
            case layout::NCDHW:   return dnnl::memory::format_tag::ncdhw;
            case layout::NDHWC:   return dnnl::memory::format_tag::ndhwc;
            case layout::TNC:     return dnnl::memory::format_tag::tnc;
            case layout::NTC:     return dnnl::memory::format_tag::ntc;
            case layout::STRIDED: return dnnl::memory::format_tag::nChw8c;
            default:              return dnnl::memory::format_tag::undef;
        }
    }

    layout cum_format_tag(dnnl::memory::format_tag format)
    {
        switch(format)
        {
            case dnnl::memory::format_tag::any:     return layout::ANY;
            case dnnl::memory::format_tag::x:       return layout::X;
            case dnnl::memory::format_tag::nc:      return layout::NC;
            // case dnnl::memory::format_tag::oi:      return layout::OI;
            case dnnl::memory::format_tag::io:      return layout::IO;
            case dnnl::memory::format_tag::nchw:    return layout::NCHW;
            case dnnl::memory::format_tag::nhwc:    return layout::NHWC;
            // case dnnl::memory::format_tag::oihw:    return layout::OIHW;
            case dnnl::memory::format_tag::hwio:    return layout::HWIO;
            case dnnl::memory::format_tag::ncdhw:   return layout::NCDHW;
            case dnnl::memory::format_tag::ndhwc:   return layout::NDHWC;
            case dnnl::memory::format_tag::tnc:     return layout::TNC;
            case dnnl::memory::format_tag::ntc:     return layout::NTC;
            case dnnl::memory::format_tag::nChw8c:  return layout::STRIDED;
            default:                                return layout::UNDEF;
        }
    }
}