/*
 * @author: jaro
 * @name:   elementwise_diffs
 * @file:   modules/cum/include/cum/neural_primitives/elementwise_diffs.hpp
 * @date:   15 September 2026 23:31:52
 */

#pragma once

#include "cum/neural_primitives/opaque_types.hpp"
#include "cum/neural_primitives/Memory.hpp"
#include "cum/neural_primitives/Descriptor.hpp"

#include "cum/functions/function_id.hpp"

#include "cum/neural_primitives/elementwise.hpp"

#include "cum/Core.hpp"

namespace cum::neural_primitives
{
    __event__ eltwise_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc,  functions::function_id algorithm, prop_kind prop_kind);
    __event__ eltwise_diff(handles::__memory__& memory, const handles::__desc__& desc,  functions::function_id algorithm, prop_kind prop_kind);

    __event__ relu_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ relu_diff(handles::__memory__& src, const handles::__desc__& desc);

    __event__ sigmoid_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ sigmoid_diff(handles::__memory__& src, const handles::__desc__& desc);

    __event__ tanh_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ tanh_diff(handles::__memory__& src, const handles::__desc__& desc);

    __event__ softmax_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ softmax_diff(handles::__memory__& src, const handles::__desc__& desc);

    __event__ gelu_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ gelu_diff(handles::__memory__& src, const handles::__desc__& desc);

    __event__ elu_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ elu_diff(handles::__memory__& src, const handles::__desc__& desc);

    __event__ leaky_relu_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc, cumeric_t alpha);
    __event__ leaky_relu_diff(handles::__memory__& src, const handles::__desc__& desc, cumeric_t alpha);

    __event__ sqrt_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc);
    __event__ sqrt_diff(handles::__memory__& src, const handles::__desc__& desc);

    __event__ square_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc);
    __event__ square_diff(handles::__memory__& src, const handles::__desc__& desc);

    /* RsrcII wrappers overloads */
    __event__ eltwise_diff(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc, functions::function_id algorithm, prop_kind prop_kind);

    __event__ relu_diff(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ relu_diff(Memory& src, const Descriptor& desc);

    __event__ sigmoid_diff(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ sigmoid_diff(Memory& src, const Descriptor& desc);

    __event__ tanh_diff(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ tanh_diff(Memory& src, const Descriptor& desc);

    __event__ softmax_diff(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ softmax_diff(Memory& src, const Descriptor& desc);

    __event__ gelu_diff(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ gelu_diff(Memory& src, const Descriptor& desc);

    __event__ elu_diff(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ elu_diff(Memory& src, const Descriptor& desc);

    __event__ leaky_relu_diff(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc, cumeric_t alpha);
    __event__ leaky_relu_diff(Memory& src, const Descriptor& desc, cumeric_t alpha);

    __event__ sqrt_diff(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc);
    __event__ sqrt_diff(Memory& src, const Descriptor& desc);

    /* Objective tensors overloads */
    __event__ eltwise_diff(Tensor& dst, const Tensor& src, functions::function_id algorithm, prop_kind prop_kind);
    __event__ eltwise_diff(Tensor& tensor, functions::function_id algorithm, prop_kind prop_kind);

    __event__ relu_diff(Tensor& dst, const Tensor& src);
    __event__ sigmoid_diff(Tensor& dst, const Tensor& src);

    __event__ tanh_diff(Tensor& dst, const Tensor& src);
    __event__ tanh_diff(Tensor& src);

    __event__ softmax_diff(Tensor& dst, const Tensor& src);
    __event__ softmax_diff(Tensor& src);

    __event__ gelu_diff(Tensor& dst, const Tensor& src);
    __event__ gelu_diff(Tensor& src);

    __event__ elu_diff(Tensor& dst, const Tensor& src);
    __event__ elu_diff(Tensor& src);

    __event__ leaky_relu_diff(Tensor& dst, const Tensor& src, cumeric_t alpha);
    __event__ leaky_relu_diff(Tensor& src, cumeric_t alpha);

    __event__ sqrt_diff(Tensor& dst, const Tensor& src);
    __event__ sqrt_diff(Tensor& src);
}