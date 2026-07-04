#include "cum/neural_primitives.hpp"

#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <stdexcept>

namespace cum::neural_primitives
{
    /* Opaque struct definitions */
    struct Memory { dnnl::memory memory; };
    struct Engine { dnnl::engine engine; };
    struct Stream { dnnl::stream stream; };

    /* Non public conversion helper */
    dnnl::memory::data_type dnnl_data_type(tensor_descriptor::data_type dtype)
    {
        switch(dtype)
        {
            case tensor_descriptor::data_type::FP64: return dnnl::memory::data_type::f64;
            case tensor_descriptor::data_type::FP32: return dnnl::memory::data_type::f32;
            case tensor_descriptor::data_type::FP16: return dnnl::memory::data_type::f16;
            case tensor_descriptor::data_type::BF16: return dnnl::memory::data_type::bf16;
            case tensor_descriptor::data_type::S8: return dnnl::memory::data_type::s8;
            case tensor_descriptor::data_type::U8: return dnnl::memory::data_type::u8;
            default: return dnnl::memory::data_type::undef;
        }
    }

    /* Non public conversion helper */
    dnnl::memory::format_tag dnnl_format_tag(tensor_descriptor::layout format)
    {
        switch(format)
        {
            case tensor_descriptor::layout::ANY: return dnnl::memory::format_tag::any;
            case tensor_descriptor::layout::X: return dnnl::memory::format_tag::x;
            case tensor_descriptor::layout::NC: return dnnl::memory::format_tag::nc;
            case tensor_descriptor::layout::OI: return dnnl::memory::format_tag::oi;
            case tensor_descriptor::layout::NCHW: return dnnl::memory::format_tag::nchw;
            case tensor_descriptor::layout::NHWC: return dnnl::memory::format_tag::nhwc;
            case tensor_descriptor::layout::STRIDED: return dnnl::memory::format_tag::nChw8c;
            default: return dnnl::memory::format_tag::undef;
        }
    }

    /* Internal tensor_descriptor implementation  */
    struct tensor_descriptor::Impl { dnnl::memory::desc desc; };

    tensor_descriptor::tensor_descriptor(tensor_descriptor::dimensions dims, tensor_descriptor::data_type dtype, tensor_descriptor::layout format)
    {
        impl = new tensor_descriptor::Impl;
        impl->desc = dnnl::memory::desc(dims, dnnl_data_type(dtype), dnnl_format_tag(format));
    }

    tensor_descriptor::~tensor_descriptor()
    {
        delete impl;
    }


}

