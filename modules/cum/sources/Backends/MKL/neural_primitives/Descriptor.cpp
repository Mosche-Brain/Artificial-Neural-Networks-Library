//
// Created by jaro on 7/19/26.
//


#include <oneapi/dnnl/dnnl.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>

#include "cum/detail/vendor/oneapi/opaque_types.hpp"
#include "cum/detail/vendor/oneapi/conversion_helpers.hpp"

#include "cum/neural_primitives/Descriptor.hpp"

namespace cum::neural_primitives
{


    Descriptor::Descriptor(Shape dims, datatype dtype, layout format) : fmt(format)
    {
        handle_ = std::make_unique<handles::__desc__>(
            dnnl::memory::desc(dims, dnnl_data_type(dtype), dnnl_format_tag(format))
        );
    }

    Descriptor::~Descriptor()
    {
    }

    dim_t Descriptor::size() const
    {
        return handle_->desc.get_size();
    }

    dim_t Descriptor::ndims() const
    {
        return handle_->desc.get_ndims();
    }

    Shape Descriptor::shape() const
    {
        return handle_->desc.get_dims();
    }

    datatype Descriptor::type() const
    {
        return cum_data_type(handle_->desc.get_data_type());
    }

    const handles::__desc__& Descriptor::handle() const
    {
        return *handle_;
    }

    handles::__desc__& Descriptor::handle()
    {
        return *handle_;
    }

    layout Descriptor::format() const
    {
        return fmt;
        // return cum_format_tag(handle_->desc.get_format_kind());
    }


} // cum