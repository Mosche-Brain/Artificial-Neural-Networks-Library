/*
 * @author: jaro
 * @name:   safetensors
 * @file:   include/yann/utils/safetensors.hpp
 * @date:   16 September 2026 21:02:45
 * @desc:   Stolen (used in compliance with the MIT license) implementation of Hugging Face safetensors format
 */

#ifndef YANN_SAFETENSORS_HPP
#define YANN_SAFETENSORS_HPP

#include <cum/datatypes.hpp>
#include <cum/Tensor.hpp>

namespace yann::utils
{
    struct TensorInfo
    {
        cum::datatype dtype;
        cum::Shape shape;
        std::array<size_t, 2> data_offsets;
    };
}

#endif //YANN_SAFETENSORS_HPP
