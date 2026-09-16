/*
 * @author: jaro
 * @name:   EltwiseKernel
 * @file:   modules/cum/include/cum/neural_primitives/EltwiseKernel.hpp
 * @date:   15 September 2026 23:39:13
 */

#pragma once

#include <memory>

#include "cum/neural_primitives/elementwise.hpp"

namespace cum::neural_primitives
{
    class EltwiseKernel
    {
    public:
        EltwiseKernel(const Descriptor& desc, functions::function_id algorithm);

        __event__ warmup();

        __event__ forward(Tensor& dst, const Tensor& src);
        __event__ forward(Tensor& src);

        __event__ forward(Memory& dst, const Memory& src);
        __event__ forward(Memory& src);

        __event__ backward(Tensor& dst, const Tensor& src);
        __event__ backward(Tensor& src);

        __event__ backward(Memory& dst, const Memory& src);
        __event__ backward(Memory& src);

        // __event__ operator()(Memory& dst, const Memory& src);
        // __event__ operator()(Memory& src);

    private:
        __event__ forward(handles::__memory__& dst, const handles::__memory__& src);

        __event__ backward(handles::__memory__& dst, const handles::__memory__& src);

        functions::function_id algorithm_;
        std::unique_ptr<Descriptor> desc;
    };
} // cum

