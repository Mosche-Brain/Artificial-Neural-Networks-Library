/*
 * @author: jaro
 * @name:   common
 * @file:   include/yann/common.hpp
 * @date:   11 November 2026 11:12:25
 */

#pragma once

#include <cum/Tensor.hpp>
#include <cum/neural_primitives/Primitive.hpp>

namespace yann
{
    using cum::Tensor;
    using cum::Matrix;
    using cum::Vector;
    using cum::DEVICE;
    using cum::neural_primitives::Primitive;
    using cum::neural_primitives::PrimitiveDescriptor;
}

namespace yann::primitives
{
    using namespace  cum::neural_primitives;
}