/*
 * @author: jaro
 * @name:   Normalize
 * @file:   include/yann/models/layers/Normalize.cpp
 * @date:   26 September 2026 09:03:40
 */

#include <stdexcept>
#include "yann/models/layers/Normalize.hpp"

namespace yann::models::layers
{
    cum::Tensor Normalize::forward(const cum::Tensor& input)
    {
        throw std::logic_error("Normalize::forward not yet implemented");
    }

    cum::Tensor Normalize::backward(const cum::Tensor& grad)
    {
        throw std::logic_error("Normalize::backward not yet implemented");
    }
} // yann