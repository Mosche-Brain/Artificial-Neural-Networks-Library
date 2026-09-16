/*
 * @author: jaro
 * @name:   elementwise_diffs
 * @file:   modules/cum/sources/Backends/MKL/neural_primitives/elementwise_diffs.cpp
 * @date:   16 September 2026 21:21:11
 */

#include <oneapi/dnnl/dnnl.hpp>
#include <stdexcept>

#include "cum/Tensor.hpp"
#include "cum/detail/vendor/oneapi/conversion_helpers.hpp"
#include "cum/detail/vendor/oneapi/event_handler.hpp"
#include "cum/detail/vendor/oneapi/opaque_types.hpp"

#include "internal/context.hpp"

#include "cum/neural_primitives/elementwise_diffs.hpp"

namespace cum::neural_primitives
{

    __event__ dnnl_eltwise_diff(handles::__memory__& dst,const handles::__memory__& src,const handles::__desc__& dst_desc, const handles::__desc__& src_desc,dnnl::algorithm algorithm,prop_kind kind,cumeric_t alpha = 0)
    {
        const dnnl::eltwise_forward::primitive_desc forward_desc = dnnl::eltwise_forward::primitive_desc(
            internal::engine(), dnnl::prop_kind::forward_training,
            algorithm,
            src_desc.desc,
            src_desc.desc,
            alpha
        );


        const dnnl::eltwise_backward backward_desc = dnnl::eltwise_backward::primitive_desc(
            internal::engine(),
            algorithm,
            dst_desc.desc,
            dst_desc.desc,
            src_desc.desc,
            alpha,
            0,
            forward_desc
            );

        const dnnl::eltwise_backward primitive = dnnl::eltwise_backward(backward_desc);
        sycl::event event = dnnl::sycl_interop::execute(
            primitive, internal::stream(),
            {
                {DNNL_ARG_SRC, src.memory},
                {DNNL_ARG_DIFF_DST, dst.memory},
                {DNNL_ARG_DIFF_SRC, dst.memory}
            });
        internal::stream().wait();
        return detail::event_handler::create(std::move(event));
    }

    __event__ dnnl_softmax_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc)
    {
        const auto axis = static_cast<int>(src_desc.desc.get_ndims() - 1);

        const auto forward_desc = dnnl::softmax_forward::primitive_desc(
            internal::engine(),
            dnnl::prop_kind::forward_training,
            dnnl::algorithm::softmax_accurate,
            src_desc.desc,
            src_desc.desc,
            axis
        );

        const dnnl::softmax_backward::primitive_desc primitive_desc = dnnl::softmax_backward::primitive_desc(
            internal::engine(),
            dnnl::algorithm::softmax_accurate,
            dst_desc.desc,
            dst_desc.desc,
            src_desc.desc,
            axis,
            forward_desc
        );

        const dnnl::softmax_backward primitive = dnnl::softmax_backward(primitive_desc);

        sycl::event event = dnnl::sycl_interop::execute(
            primitive, internal::stream(),
            {
                {DNNL_ARG_DST, src.memory},
                {DNNL_ARG_DIFF_DST, dst.memory},
                {DNNL_ARG_DIFF_SRC, dst.memory}
            }
        );

        internal::stream().wait();
        return detail::event_handler::create(std::move(event));
    }

    __event__ dispatch_diff(handles::__memory__& dst,
                            const handles::__memory__& src,
                            const handles::__desc__& dst_desc,
                            const handles::__desc__& src_desc,
                            functions::function_id algorithm,
                            prop_kind kind,
                            cumeric_t alpha = 0)
    {
        if(algorithm == functions::function_id::undefined)
            throw std::invalid_argument("Eltwise derivative requires a known algorithm");
        return dnnl_eltwise_diff(dst, src, dst_desc, src_desc, dnnl_algorithm(algorithm), kind, alpha);
    }


    __event__ eltwise_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc, functions::function_id algorithm, prop_kind kind)
    {
        return dispatch_diff(dst, src, dst_desc, src_desc, algorithm, kind);
    }


    __event__ eltwise_diff(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc, functions::function_id algorithm, prop_kind kind)
    {
        return eltwise_diff(dst.handle(), src.handle(), dst_desc.handle(), src_desc.handle(), algorithm, kind);
    }

    __event__ eltwise_diff(Memory& memory, const Descriptor& desc, functions::function_id algorithm, prop_kind kind)
    {
        return eltwise_diff(memory.handle(), desc.handle(), algorithm, kind);
    }
    __event__ eltwise_diff(handles::__memory__& memory, const handles::__desc__& desc, functions::function_id algorithm, prop_kind kind)
    {
        return eltwise_diff(memory, memory, desc, desc, algorithm, kind);
    }

    #define CUM_DIFF_RAW(NAME, ID) \
    \
    __event__ NAME(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc) \
    { \
        return dispatch_diff(dst, src, dst_desc, src_desc, ID, prop_kind::forward); \
    } \
    __event__ NAME(handles::__memory__& src, const handles::__desc__& desc) \
    { \
        return NAME(src, src, desc, desc); \
    } \

    CUM_DIFF_RAW(relu_diff, functions::function_id::relu)
    CUM_DIFF_RAW(sigmoid_diff, functions::function_id::sigmoid)
    CUM_DIFF_RAW(tanh_diff, functions::function_id::tanh)
    CUM_DIFF_RAW(gelu_diff, functions::function_id::gelu)
    CUM_DIFF_RAW(elu_diff, functions::function_id::elu)
    CUM_DIFF_RAW(sqrt_diff, functions::function_id::sqrt)
    __event__ square_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc) \
    { \
        return dnnl_eltwise_diff(dst, src, dst_desc, src_desc, dnnl::algorithm::eltwise_square, prop_kind::forward);
    } \
    __event__ square_diff(handles::__memory__& src, const handles::__desc__& desc) \
    { \
        return square_diff(src, src, desc, desc); \
    } \

    #undef CUM_DIFF_RAW

    __event__ softmax_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc)
    {
        return dnnl_softmax_diff(dst, src, dst_desc, src_desc);
    }

    __event__ softmax_diff(handles::__memory__& src, const handles::__desc__& desc)
    {
        return softmax_diff(src, src, desc, desc);
    }

    __event__ leaky_relu_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc, cumeric_t alpha)
    {
        return dispatch_diff(dst, src, dst_desc, src_desc, functions::function_id::leaky_relu, prop_kind::forward, alpha);
    }

    __event__ leaky_relu_diff(handles::__memory__& src, const handles::__desc__& desc, cumeric_t alpha)
    {
        return leaky_relu_diff(src, src, desc, desc, alpha);
    }

#define CUM_DIFF_MEMORY(NAME, RAW) \
    \
    __event__ NAME(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc) \
    { \
        return RAW(dst.handle(), src.handle(), dst_desc.handle(), src_desc.handle()); \
    } \
    __event__ NAME(Memory& src, const Descriptor& desc) \
    {  \
        return RAW(src.handle(), desc.handle()); \
    } \

    CUM_DIFF_MEMORY(relu_diff, relu_diff);
    CUM_DIFF_MEMORY(sigmoid_diff, sigmoid_diff);
    CUM_DIFF_MEMORY(tanh_diff, tanh_diff);
    CUM_DIFF_MEMORY(softmax_diff, softmax_diff);
    CUM_DIFF_MEMORY(gelu_diff, gelu_diff);
    CUM_DIFF_MEMORY(elu_diff, elu_diff);
    CUM_DIFF_MEMORY(sqrt_diff, sqrt_diff);
    CUM_DIFF_MEMORY(square_diff, square_diff);
#undef CUM_DIFF_MEMORY

    __event__ leaky_relu_diff(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc, cumeric_t alpha)
    {
        return leaky_relu_diff(dst.handle(), src.handle(), dst_desc.handle(), src_desc.handle(), alpha);
    }
    __event__ leaky_relu_diff(Memory& src, const Descriptor& desc, cumeric_t alpha)
    {
        return leaky_relu_diff(src.handle(), desc.handle(), alpha);
    }

#define CUM_DIFF_TENSOR(NAME, MEMORY) \
    \
    __event__ NAME(Tensor& dst, const Tensor& src) \
    { \
        return MEMORY(*dst.memory(), *src.memory(), *dst.descriptor(), *src.descriptor()); \
    } \
    __event__ NAME(Tensor& src) \
    { \
        return MEMORY(*src.memory(), *src.descriptor()); \
    } \


    CUM_DIFF_TENSOR(relu_diff, relu_diff);
    CUM_DIFF_TENSOR(sigmoid_diff, sigmoid_diff);
    CUM_DIFF_TENSOR(tanh_diff, tanh_diff);
    CUM_DIFF_TENSOR(softmax_diff, softmax_diff);
    CUM_DIFF_TENSOR(gelu_diff, gelu_diff);
    CUM_DIFF_TENSOR(elu_diff, elu_diff);
    CUM_DIFF_TENSOR(sqrt_diff, sqrt_diff);
    CUM_DIFF_TENSOR(square_diff, square_diff);
 #undef CUM_DIFF_TENSOR

    __event__ eltwise_diff(Tensor& dst, const Tensor& src, functions::function_id algorithm, prop_kind kind)
    {
        return eltwise_diff(*dst.memory(), *src.memory(), *dst.descriptor(), *src.descriptor(), algorithm, kind);
    }

    __event__ eltwise_diff(Tensor& tensor, functions::function_id algorithm, prop_kind kind)
    {
        return eltwise_diff(*tensor.memory(), *tensor.descriptor(), algorithm, kind);
    }

    __event__ leaky_relu_diff(Tensor& dst, const Tensor& src, cumeric_t alpha)
    {
        return leaky_relu_diff(*dst.memory(), *src.memory(), *dst.descriptor(), *src.descriptor(), alpha);
    }

    __event__ leaky_relu_diff(Tensor& src, cumeric_t alpha)
    {
        return leaky_relu_diff(*src.memory(), *src.descriptor(), alpha);
    }
}
