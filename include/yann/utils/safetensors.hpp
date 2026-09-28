/*
 * @author: jaro
 * @name:   safetensors
 * @file:   include/yann/utils/safetensors.hpp
 * @date:   16 September 2026 21:02:45
 * @desc:   Stolen (used in compliance with the MIT license) implementation of Hugging Face safetensors format
 */

#ifndef YANN_SAFETENSORS_HPP
#define YANN_SAFETENSORS_HPP

#pragma once


#include <unordered_map>

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

    using SafeTensors = std::unordered_map<std::string, cum::Tensor>;

    std::unordered_map<std::string, cum::Tensor> load_safetensors(const std::string &filename);

    void save_safetensors(const std::unordered_map<std::string, cum::Tensor> &tensors, const std::string &filename, const std::unordered_map<std::string, std::string> &metadata = {});

}

#endif //YANN_SAFETENSORS_HPP
