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

    Descriptor::Descriptor(Descriptor& other) : fmt(other.fmt)
    {
        handle_ = std::make_unique<handles::__desc__>(other.handle_->desc);
    }

    Descriptor::Descriptor(const handles::__desc__& handle) : fmt(cum::layout::ANY)
    {
        handle_ = std::make_unique<handles::__desc__>(handle);
    }

    Descriptor::Descriptor(const handles::__desc__& handle, layout format) : fmt(format)
    {
        handle_ = std::make_unique<handles::__desc__>(handle);
    }

    Descriptor::Descriptor(Descriptor&&) noexcept = default;
    Descriptor& Descriptor::operator=(Descriptor&&) noexcept = default;

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

    dim_t Descriptor::stride() const
    {
        return 69;
        // return handle_->desc.get_strides();
    }

    dim_t Descriptor::offset() const
    {
        return handle_->desc.get_submemory_offset();
    }


    Descriptor Descriptor::reshape(const Shape& shape)
    {
        // return Descriptor(shape, type(), format());
        return Descriptor({handle_->desc.reshape(shape)}, format());
    }

    Descriptor& Descriptor::reshape_in_place(const Shape& shape)
    {
        // Check is the sum of elements in old and new shape equak
        size_t sum_a = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        size_t sum_b = std::accumulate(handle_->desc.get_dims().begin(), handle_->desc.get_dims().end(), 1, std::multiplies<size_t>());

        if (sum_a != sum_b) throw std::runtime_error("Cannot reshape tensor: products of old and new dimensions vector aren't equal");

        handle_ = std::make_unique<handles::__desc__>(handle_->desc.reshape(shape));
        return *this;
    }

    Descriptor& Descriptor::operator = (Descriptor& other)
    {
        if (this == &other) return *this;

        this->fmt = other.fmt;
        this->handle_ = std::make_unique<handles::__desc__>(other.handle_->desc);

        return *this;
    }

} // cum