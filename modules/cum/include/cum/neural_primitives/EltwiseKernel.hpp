/*
 * @author: jaro
 * @name:   EltwiseKernel
 * @file:   modules/cum/include/cum/neural_primitives/EltwiseKernel.hpp
 * @date:   15 September 2026 23:39:13
 */

#pragma once

#include "cum/neural_primitives/elementwise.hpp"

namespace cum::neural_primitives
{
    class EltwiseKernel
    {
    public:
        EltwiseKernel(const Descriptor& dst, const Descriptor& src, functions::function_id algorithm);

        __event__ warmup();

        __event__ forward(Memory& dst, const Memory& src);
        __event__ forward(Memory& src);

        __event__ backward(Memory& dst, const Memory& src);
        __event__ backward(Memory& src);

        // __event__ operator()(Memory& dst, const Memory& src);
        // __event__ operator()(Memory& src);

    private:
        functions::function_id algorithm_;
        Descriptor dst_desc;
        Descriptor src_desc;
    };
} // cum

