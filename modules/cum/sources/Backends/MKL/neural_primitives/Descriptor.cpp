//
// Created by jaro on 7/19/26.
//


#include <oneapi/dnnl/dnnl.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>

#include "cum/detail/vendor/oneapi/opaque_types.hpp"

#include "cum/neural_primitives/Descriptor.hpp"

namespace cum::neural_primitives
{
    dnnl::memory::data_type dnnl_data_type(datatype dtype)
    {
        switch(dtype)
        {
            case datatype::FP64: return dnnl::memory::data_type::f64;
            case datatype::FP32: return dnnl::memory::data_type::f32;
            case datatype::FP16: return dnnl::memory::data_type::f16;
            case datatype::BF16: return dnnl::memory::data_type::bf16;
            case datatype::S8:   return dnnl::memory::data_type::s8;
            case datatype::U8:   return dnnl::memory::data_type::u8;
            default: return dnnl::memory::data_type::undef;
        }
    }

    /* Non public conversion helper */
    dnnl::memory::format_tag dnnl_format_tag(layout format)
    {
        switch(format)
        {
        case layout::ANY: return dnnl::memory::format_tag::any;
        case layout::X: return dnnl::memory::format_tag::x;
        case layout::NC: return dnnl::memory::format_tag::nc;
        case layout::OI: return dnnl::memory::format_tag::oi;
        case layout::NCHW: return dnnl::memory::format_tag::nchw;
        case layout::NHWC: return dnnl::memory::format_tag::nhwc;
        case layout::STRIDED: return dnnl::memory::format_tag::nChw8c;
        default: return dnnl::memory::format_tag::undef;
        }
    }

    Descriptor::Descriptor(Shape dims, datatype dtype, layout format)
    {
        handle = std::make_unique<handles::__desc__>(
            dnnl::memory::desc(dims, dnnl_data_type(dtype), dnnl_format_tag(format))
        );
    }

    Descriptor::~Descriptor()
    {
    }
} // cum