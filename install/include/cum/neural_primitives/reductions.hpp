/*
 * @author: jaro
 * @name:   reductions
 * @file:   modules/cum/sources/Backends/MKL/neural_primitives/reductions.hpp
 * @date:   20 September 2026 16:40:44
 */

#pragma once

#include "cum/neural_primitives/Memory.hpp"
#include "cum/neural_primitives/Descriptor.hpp"
#include "cum/neural_primitives/opaque_types.hpp"

#include "cum/Core.hpp"

namespace cum { class Tensor; }

namespace cum::neural_primitives
{
    /* Raw handles overloads */

    __event__ softmax(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc, dim_t axis = -1);
    __event__ softmax(handles::__memory__& memory, const handles::__desc__& desc, dim_t axis = -1);

    __event__ softmax_diff(handles::__memory__& dst, const handles::__memory__& src, const handles::__desc__& r_desc, const handles::__desc__& a_desc, dim_t axis = -1);
    __event__ softmax_diff(handles::__memory__& memory, const handles::__desc__& desc, dim_t axis = -1);

    /* RAII wrappers overloads */

    __event__ softmax(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc, dim_t axis = -1);
    __event__ softmax(Memory& src, const Descriptor& desc, dim_t axis = -1);

    __event__ softmax_diff(Memory& dst, const Memory& src, const Descriptor& r_desc, const Descriptor& a_desc, dim_t axis = -1);
    __event__ softmax_diff(Memory& memory, const Descriptor& desc, dim_t axis = -1);

    /* Objective tensors wrappers overloads */

    __event__ softmax(Tensor& dst, const Tensor& src, dim_t axis = -1);
    __event__ softmax(Tensor& tensor, dim_t axis = -1);

    __event__ softmax_diff(Tensor& dst, const Tensor& src, dim_t axis = -1);
    __event__ softmax_diff(Tensor& tensor, dim_t axis = -1);

}
