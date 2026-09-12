/*
 * @author: jaro
 * @name:   elementwise
 * @file:   modules/cum/sources/Backends/MKL/neural_primitives/elementwise.cpp
 * @date:   12 September 2026 16:48:50
 */

#include "internal/context.hpp"
#include "cum/detail/vendor/oneapi/opaque_types.hpp"
#include "cum/detail/vendor/oneapi/event_handler.hpp"

#include "cum/Tensor.hpp"

#include "cum/neural_primitives/elementwise.hpp"

namespace cum::neural_primitives
{
    // conversion helper function_id -> dnnl::algortithm
    dnnl::algorithm dnnl_algorithm(functions::function_id id)
    {
        switch (id)
        {
            case functions::function_id::relu: return dnnl::algorithm::eltwise_relu;
            case functions::function_id::sigmoid: return dnnl::algorithm::eltwise_logistic;
            case functions::function_id::tanh: return dnnl::algorithm::eltwise_tanh;
            case functions::function_id::elu: return dnnl::algorithm::eltwise_elu;
            case functions::function_id::leaky_relu: return dnnl::algorithm::eltwise_relu;
            // case functions::function_id::softmax: return dnnl::algorithm::softmax_accurate;
            case functions::function_id::gelu: return dnnl::algorithm::eltwise_gelu_tanh;
            default: return dnnl::algorithm::undef;
        }
    }

    /* Raw handles overloads */


    __event__ eltwise(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc, dnnl::algorithm algorithm, dnnl::prop_kind prop_kind) // does
    {
        dnnl::eltwise_forward::primitive_desc primitive_desc {
            internal::engine(),
            prop_kind,
            algorithm,
            src_desc.desc,
            dst_desc.desc
        };

        dnnl::primitive primitive = dnnl::eltwise_forward(primitive_desc);

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

    __event__ eltwise(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc,  functions::function_id algorithm, prop_kind prop_kind)
    {
        return eltwise(dst, src, dst_desc, src_desc, dnnl_algorithm(algorithm), prop_kind);
    }

    __event__ relu(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc)
    {
        dnnl::eltwise_forward::primitive_desc primitive_desc {
            internal::engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_relu,
            src_desc.desc,
            dst_desc.desc
        };

        dnnl::primitive primitive = dnnl::eltwise_forward(primitive_desc);

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
    __event__ relu(handles::__memory__& src, const handles::__desc__& desc)
    {

    }

    __event__ sigmoid(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc)
    {
        dnnl::eltwise_forward::primitive_desc primitive_desc {
            internal::engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_logistic,
            src_desc.desc,
            dst_desc.desc
        };

        dnnl::primitive primitive = dnnl::eltwise_forward(primitive_desc);

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
    __event__ sigmoid(handles::__memory__& src, const handles::__desc__& desc)
    {

    }

    __event__ tanh(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc)
    {
        dnnl::eltwise_forward::primitive_desc primitive_desc {
            internal::engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_tanh,
            src_desc.desc,
            dst_desc.desc
        };

        dnnl::primitive primitive = dnnl::eltwise_forward(primitive_desc);

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
    __event__ tanh(handles::__memory__& src, const handles::__desc__& desc)
    {

    }

    __event__ softmax(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc)
    {
        dnnl::softmax_forward::primitive_desc primitive_desc {
            internal::engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::softmax_accurate,
            src_desc.desc,
            dst_desc.desc,
            1
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
    __event__ softmax(handles::__memory__& src, const handles::__desc__& desc)
    {

    }

    __event__ gelu(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc)
    {
        dnnl::eltwise_forward::primitive_desc primitive_desc {
            internal::engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_gelu_tanh,
            src_desc.desc,
            dst_desc.desc
        };

        dnnl::primitive primitive = dnnl::eltwise_forward(primitive_desc);

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
    __event__ gelu(handles::__memory__& src, const handles::__desc__& desc)
    {

    }

    __event__ elu(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc)
    {
        dnnl::eltwise_forward::primitive_desc primitive_desc {
            internal::engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_elu,
            src_desc.desc,
            dst_desc.desc
        };

        dnnl::primitive primitive = dnnl::eltwise_forward(primitive_desc);

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
    __event__ elu(handles::__memory__& src, const handles::__desc__& desc)
    {

    }

    __event__ leaky_relu(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc, const cumeric_t alpha)
    {
        dnnl::eltwise_forward::primitive_desc primitive_desc {
            internal::engine(),
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_relu,
            src_desc.desc,
            dst_desc.desc,
            alpha
        };

        dnnl::primitive primitive = dnnl::eltwise_forward(primitive_desc);

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
    __event__ leaky_relu(handles::__memory__& src, const handles::__desc__& desc)
    {

    }

    /* RAII wrappers overloads */

    __event__ relu(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc)
    {
        handles::__memory__& dst_memory = dst.handle();
        const handles::__memory__& src_memory = src.handle();

        const handles::__desc__& dst_descriptor = dst_desc.handle();
        const handles::__desc__& src_descriptor = src_desc.handle();

        return relu(dst_memory, src_memory, dst_descriptor, src_descriptor);
    }
    __event__ relu(Memory& src, const Descriptor& desc)
    {

    }

    __event__ sigmoid(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc)
    {
        handles::__memory__& dst_memory = dst.handle();
        const handles::__memory__& src_memory = src.handle();

        const handles::__desc__& dst_descriptor = dst_desc.handle();
        const handles::__desc__& src_descriptor = src_desc.handle();

        return sigmoid(dst_memory, src_memory, dst_descriptor, src_descriptor);
    }
    __event__ sigmoid(Memory& src, const Descriptor& desc)
    {

    }

    __event__ tanh(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc)
    {
        handles::__memory__& dst_memory = dst.handle();
        const handles::__memory__& src_memory = src.handle();

        const handles::__desc__& dst_descriptor = dst_desc.handle();
        const handles::__desc__& src_descriptor = src_desc.handle();

        return tanh(dst_memory, src_memory, dst_descriptor, src_descriptor);
    }
    __event__ tanh(Memory& src, const Descriptor& desc)
    {

    }

    __event__ softmax(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc)
    {
        handles::__memory__& dst_memory = dst.handle();
        const handles::__memory__& src_memory = src.handle();

        const handles::__desc__& dst_descriptor = dst_desc.handle();
        const handles::__desc__& src_descriptor = src_desc.handle();

        return softmax(dst_memory, src_memory, dst_descriptor, src_descriptor);
    }
    __event__ softmax(Memory& src, const Descriptor& desc)
    {

    }

    __event__ gelu(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc)
    {
        handles::__memory__& dst_memory = dst.handle();
        const handles::__memory__& src_memory = src.handle();

        const handles::__desc__& dst_descriptor = dst_desc.handle();
        const handles::__desc__& src_descriptor = src_desc.handle();

        return gelu(dst_memory, src_memory, dst_descriptor, src_descriptor);
    }
    __event__ gelu(Memory& src, const Descriptor& desc)
    {

    }

    __event__ elu(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc)
    {
        handles::__memory__& dst_memory = dst.handle();
        const handles::__memory__& src_memory = src.handle();

        const handles::__desc__& dst_descriptor = dst_desc.handle();
        const handles::__desc__& src_descriptor = src_desc.handle();

        return elu(dst_memory, src_memory, dst_descriptor, src_descriptor);
    }
    __event__ elu(Memory& src, const Descriptor& desc)
    {

    }

    __event__ leaky_relu(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc, const cumeric_t alpha)
    {
        handles::__memory__& dst_memory = dst.handle();
        const handles::__memory__& src_memory = src.handle();

        const handles::__desc__& dst_descriptor = dst_desc.handle();
        const handles::__desc__& src_descriptor = src_desc.handle();

        return leaky_relu(dst_memory, src_memory, dst_descriptor, src_descriptor, alpha);
    }
    __event__ leaky_relu(Memory& src, const Descriptor& desc, const cumeric_t alpha)
    {

    }

    /* Objective tensors overloads */

    __event__ relu(Tensor& dst, const Tensor& src)
    {
        handles::__memory__& dst_memory = dst.memory()->handle();
        const handles::__memory__& src_memory = src.memory()->handle();

        const handles::__desc__& dst_descriptor = dst.descriptor()->handle();
        const handles::__desc__& a_descriptor = src.descriptor()->handle();

        return relu(dst_memory, src_memory, dst_descriptor, a_descriptor);
    }
    __event__ sigmoid(Tensor& dst, const Tensor& src)
    {

    }

    __event__ tanh(Tensor& dst, const Tensor& src)
    {

    }
    __event__ tanh(Tensor& src)
    {

    }

    __event__ softmax(Tensor& dst, const Tensor& src)
    {

    }
    __event__ softmax(Tensor& src)
    {

    }

    __event__ gelu(Tensor& dst, const Tensor& src)
    {

    }
    __event__ gelu(Tensor& src)
    {

    }

    __event__ elu(Tensor& dst, const Tensor& src)
    {

    }
    __event__ elu(Tensor& src)
    {

    }

    __event__ leaky_relu(Tensor& dst, const Tensor& src, const cumeric_t alpha)
    {

    }
    __event__ leaky_relu(Tensor& src, const cumeric_t alpha)
    {

    }
}