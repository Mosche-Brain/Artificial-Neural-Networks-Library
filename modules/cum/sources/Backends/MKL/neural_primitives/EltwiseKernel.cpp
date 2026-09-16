/*
 * @author: jaro
 * @name:   EltwiseKernel
 * @file:   modules/cum/include/cum/neural_primitives/EltwiseKernel.cpp
 * @date:   15 September 2026 23:39:13
 */

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include "internal/context.hpp"
#include "cum/detail/vendor/oneapi/conversion_helpers.hpp"
#include "cum/detail/vendor/oneapi/event_handler.hpp"
#include "cum/detail/vendor/oneapi/opaque_types.hpp"

#include "cum/Tensor.hpp"

#include "cum/neural_primitives/EltwiseKernel.hpp"

namespace cum::neural_primitives
{
    EltwiseKernel::EltwiseKernel(const Descriptor& desc, functions::function_id algorithm)
        :  algorithm_(algorithm), desc(std::make_unique<Descriptor>(desc.shape(), desc.type(), desc.format()))
    {

    }

    __event__ EltwiseKernel::forward(Tensor& dst, const Tensor& src)
    {
        return this->forward(dst.memory()->handle(), src.memory()->handle());
    }

    __event__ EltwiseKernel::forward(Memory& dst, const Memory& src)
    {
        return this->forward(dst.handle(), src.handle());
    }

    __event__ EltwiseKernel::backward(Tensor& dst, const Tensor& src)
    {
        return this->backward(dst.memory()->handle(), src.memory()->handle());
    }

    __event__ EltwiseKernel::backward(Memory& dst, const Memory& src)
    {
        return this->backward(dst.handle(), src.handle());
    }

    __event__ EltwiseKernel::forward(handles::__memory__& dst, const handles::__memory__& src)
    {
        dnnl::eltwise_forward::primitive_desc primitive_desc {
            internal::engine(),
            dnnl::prop_kind::forward_inference,
            dnnl_algorithm(algorithm_),
            desc->handle().desc,
            desc->handle().desc
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

    __event__ EltwiseKernel::backward(handles::__memory__& dst, const handles::__memory__& src)
    {
        dnnl::eltwise_forward::primitive_desc forward_desc {
            internal::engine(),
            dnnl::prop_kind::forward_training,
            dnnl_algorithm(algorithm_),
            desc->handle().desc,
            desc->handle().desc
        };

        dnnl::eltwise_backward::primitive_desc backward_desc {
            internal::engine(),
            dnnl_algorithm(algorithm_),
            desc->handle().desc,
            desc->handle().desc,
            desc->handle().desc,
            forward_desc
        };

        dnnl::primitive primitive = dnnl::eltwise_backward(backward_desc);
        sycl::event event = dnnl::sycl_interop::execute(
            primitive,
            internal::stream(),
            {
                { DNNL_ARG_SRC, src.memory },
                { DNNL_ARG_DIFF_DST, dst.memory },
                { DNNL_ARG_DIFF_SRC, dst.memory }
            }
        );

        internal::stream().wait();
        return detail::event_handler::create(std::move(event));
    }
} // cum