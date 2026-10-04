/*
 * @author: jaro
 * @name:   reductions
 * @file:   modules/cum/sources/Backends/MKL/neural_primitives/reductions.cpp
 * @date:   20 September 2026 16:40:44
 */

#include "internal/context.hpp"
#include "cum/detail/vendor/oneapi/opaque_types.hpp"
#include "cum/detail/vendor/oneapi/event_handler.hpp"
#include "cum/detail/vendor/oneapi/conversion_helpers.hpp"

#include "cum/Tensor.hpp"

#include "cum/neural_primitives/reductions.hpp"

namespace cum::neural_primitives
{
    using namespace handles;

    /* Handles overloads */
    __event__ softmax(__memory__& dst, const __memory__& src, const __desc__& dst_desc, const __desc__& src_desc, const dim_t axis)
    {
        int axis_ = axis == -1 ? src_desc.desc.get_ndims() - 1 : axis;

        dnnl::softmax_forward::primitive_desc primitive_desc {
            internal::engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::softmax_accurate,
            src_desc.desc,
            dst_desc.desc,
            axis_
        };

        dnnl::primitive primitive = dnnl::softmax_forward(primitive_desc);

        sycl::event event = dnnl::sycl_interop::execute(
            primitive,
            internal::stream(),
            {
                { DNNL_ARG_SRC, src.memory },
                { DNNL_ARG_DST, dst.memory }
            }
        );

        internal::stream().wait();
        return detail::event_handler::create(std::move(event));
    }
    __event__ softmax(__memory__& memory, const __desc__& desc, const dim_t axis)
    {
        return softmax(memory, memory, desc, desc);
    }

    __event__ softmax_diff(__memory__& dst, const __memory__& src, const __desc__& dst_desc, const __desc__& src_desc, dim_t axis)
    {
        int axis_ = axis == -1 ? src_desc.desc.get_ndims() - 1 : axis;

        dnnl::softmax_forward::primitive_desc forward_desc {
            internal::engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::softmax_accurate,
            src_desc.desc,
            dst_desc.desc,
            axis_
        };

        dnnl::softmax_backward backward_desc = dnnl::softmax_backward::primitive_desc(
            internal::engine(),
            dnnl::algorithm::softmax_accurate,
            dst_desc.desc,
            dst_desc.desc,
            src_desc.desc,
            axis_,
            forward_desc
        );

        dnnl::primitive primitive = dnnl::softmax_forward(forward_desc);

        sycl::event event = dnnl::sycl_interop::execute(
            primitive,
            internal::stream(),
            {
                { DNNL_ARG_SRC, src.memory },
                { DNNL_ARG_DST, dst.memory }
            }
        );

        internal::stream().wait();
        return detail::event_handler::create(std::move(event));
    }

    __event__ softmax_diff(__memory__& memory, const __desc__& desc, const dim_t axis)
    {
        return softmax_diff(memory, memory, desc, desc);
    }

    /* RAII overloads */

    __event__ softmax(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc, const dim_t axis)
    {
        __memory__& dst_memory = dst.handle();
        const __memory__& src_memory = src.handle();

        const __desc__& dst_descriptor = dst_desc.handle();
        const __desc__& src_descriptor = src_desc.handle();

        return softmax(dst_memory, src_memory, dst_descriptor, src_descriptor, axis);
    }

    __event__ softmax(Memory& src, const Descriptor& desc, const dim_t axis)
    {
        return softmax(src, src, desc, desc, axis);
    }

    __event__ softmax_diff(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc, const dim_t axis)
    {
        __memory__& dst_memory = dst.handle();
        const __memory__& src_memory = src.handle();

        const __desc__& dst_descriptor = dst_desc.handle();
        const __desc__& src_descriptor = src_desc.handle();

        return softmax_diff(dst_memory, src_memory, dst_descriptor, src_descriptor, axis);
    }

    __event__ softmax_diff(Memory& memory, const Descriptor& desc, const dim_t axis)
    {
        return softmax_diff(memory, memory, desc, desc, axis);
    }

    /* Tensor overloads */

    __event__ softmax(Tensor& dst, const Tensor& src, const dim_t axis)
    {
        __memory__& dst_memory = dst.memory()->handle();
        const __memory__& src_memory = src.memory()->handle();

        const __desc__& dst_descriptor = dst.descriptor()->handle();
        const __desc__& src_descriptor = src.descriptor()->handle();

        return softmax(dst_memory, src_memory, dst_descriptor, src_descriptor, axis);
    }

    __event__ softmax(Tensor& tensor, const dim_t axis)
    {
        __memory__& memory = tensor.memory()->handle();
        const __desc__& descriptor = tensor.descriptor()->handle();

        return softmax(memory, memory, descriptor, descriptor, axis);
    }

    __event__ softmax_diff(Tensor& dst, const Tensor& src, const dim_t axis)
    {
        __memory__& dst_memory = dst.memory()->handle();
        const __memory__& src_memory = src.memory()->handle();

        const __desc__& dst_descriptor = dst.descriptor()->handle();
        const __desc__& src_descriptor = src.descriptor()->handle();

        return softmax_diff(dst_memory, src_memory, dst_descriptor, src_descriptor, axis);
    }

    __event__ softmax_diff(Tensor& tensor, const dim_t axis)
    {
        __memory__& memory = tensor.memory()->handle();
        const __desc__& descriptor = tensor.descriptor()->handle();

        return softmax_diff(memory, descriptor, axis);
    }
}