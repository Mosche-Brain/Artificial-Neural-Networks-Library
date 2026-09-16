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

namespace cum::neural_primitives::diff
{
    __event__ eltwise(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc,  functions::function_id algorithm, prop_kind prop_kind);
    __event__ eltwise(handles::__memory__& memory, const handles::__desc__& desc,  functions::function_id algorithm, prop_kind prop_kind);

    __event__ relu(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ relu(handles::__memory__& src, const handles::__desc__& desc);

    __event__ sigmoid(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ sigmoid(handles::__memory__& src, const handles::__desc__& desc);

    __event__ tanh(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ tanh(handles::__memory__& src, const handles::__desc__& desc);

    __event__ softmax(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ softmax(handles::__memory__& src, const handles::__desc__& desc);

    __event__ gelu(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ gelu(handles::__memory__& src, const handles::__desc__& desc);

    __event__ elu(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc);
    __event__ elu(handles::__memory__& src, const handles::__desc__& desc);

    __event__ leaky_relu(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc, cumeric_t alpha);
    __event__ leaky_relu(handles::__memory__& src, const handles::__desc__& desc, cumeric_t alpha);

    __event__ sqrt(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc);
    __event__ sqrt(handles::__memory__& src, const handles::__desc__& desc);

    __event__ square(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc);
    __event__ square(handles::__memory__& src, const handles::__desc__& desc);

    /* RsrcII wrappers overloads */
    __event__ eltwise(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc, functions::function_id algorithm, prop_kind prop_kind);

    __event__ relu(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ relu(Memory& src, const Descriptor& desc);

    __event__ sigmoid(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ sigmoid(Memory& src, const Descriptor& desc);

    __event__ tanh(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ tanh(Memory& src, const Descriptor& desc);

    __event__ softmax(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ softmax(Memory& src, const Descriptor& desc);

    __event__ gelu(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ gelu(Memory& src, const Descriptor& desc);

    __event__ elu(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc);
    __event__ elu(Memory& src, const Descriptor& desc);

    __event__ leaky_relu(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc, cumeric_t alpha);
    __event__ leaky_relu(Memory& src, const Descriptor& desc, cumeric_t alpha);

    __event__ sqrt(Memory& dst, const Memory& src, const Descriptor& dst_desc, const Descriptor& src_desc);
    __event__ sqrt(Memory& src, const Descriptor& desc);

    /* Objective tensors overloads */
    __event__ eltwise(Tensor& dst, const Tensor& src, functions::function_id algorithm, prop_kind prop_kind);
    __event__ eltwise(Tensor& tensor, functions::function_id algorithm, prop_kind prop_kind);

    __event__ relu(Tensor& dst, const Tensor& src);
    __event__ sigmoid(Tensor& dst, const Tensor& src);

    __event__ tanh(Tensor& dst, const Tensor& src);
    __event__ tanh(Tensor& src);

    __event__ softmax(Tensor& dst, const Tensor& src);
    __event__ softmax(Tensor& src);

    __event__ gelu(Tensor& dst, const Tensor& src);
    __event__ gelu(Tensor& src);

    __event__ elu(Tensor& dst, const Tensor& src);
    __event__ elu(Tensor& src);

    __event__ leaky_relu(Tensor& dst, const Tensor& src, cumeric_t alpha);
    __event__ leaky_relu(Tensor& src, cumeric_t alpha);

    __event__ sqrt(Tensor& dst, const Tensor& src);
    __event__ sqrt(Tensor& src);
}