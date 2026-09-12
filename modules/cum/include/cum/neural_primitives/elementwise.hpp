/*
 * @author: jaro
 * @name:   elementwise
 * @file:   modules/cum/include/cum/neural_primitives/elementwise.hpp
 * @date:   12 September 2026 16:40:14
 */

#pragma once

#include "cum/neural_primitives/opaque_types.hpp"
#include "cum/neural_primitives/Memory.hpp"
#include "cum/neural_primitives/Descriptor.hpp"

#include "cum/functions/function_id.hpp"

#include "cum/Core.hpp"

namespace cum { class Tensor; }

namespace cum::neural_primitives
{
    enum class prop_kind { // temporalery copied directly form dnnl.hpp
        /// Undefined propagation kind.
        undef = 0,
        /// Forward data propagation (training mode). In this mode, primitives
            /// perform computations necessary for subsequent backward propagation.
        forward_training = 64,
        /// Forward data propagation (inference mode). In this mode, primitives
            /// perform only computations that are necessary for inference and omit
            /// computations that are necessary only for backward propagation.
        forward_inference = 96,
        /// Forward data propagation,
            /// alias for #dnnl::prop_kind::forward_training.
        forward = 64,
        /// Backward propagation (with respect to all parameters).
        backward = 128,
        /// Backward data propagation.
        backward_data = 160,
        /// Backward weights propagation.
        backward_weights = 192,
        /// Backward bias propagation.
        backward_bias = 192
    };

    /* Raw handles overloads */
    __event__ eltwise(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& dst_desc, const handles::__desc__& src_desc,  functions::function_id algorithm, prop_kind prop_kind);

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

    /* Objective tensors overloads */
    __event__ eltwise(Tensor& dst, const Tensor& src, functions::function_id algorithm, prop_kind prop_kind);

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

}