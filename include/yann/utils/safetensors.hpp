/**
 * @file safetensors.hpp
 * @author jaro
 * @brief Hugging Face safetensors format reader and writer for YANN tensors.
 * @date 2026-09-16
 * @details Re-exports and wraps the safetensors implementation provided by the safetensors.cpp submodule.
 */

#ifndef YANN_SAFETENSORS_HPP
#define YANN_SAFETENSORS_HPP

#pragma once

#include <safetensors/safetensors.hpp>

namespace yann::utils
{
    /**
     * @brief Metadata describing a single tensor serialized in a safetensors file.
     */
    using TensorInfo = ::safetensors::TensorInfo;

    /**
     * @brief Type alias for a map of named tensors loaded from/saved to safetensors format.
     */
    using SafeTensors = ::safetensors::SafeTensors;

    /**
     * @brief Load tensors from a .safetensors file into a map of named cum::Tensor objects.
     */
    using ::safetensors::load_safetensors;

    /**
     * @brief Save a map of named tensors to a .safetensors file.
     */
    using ::safetensors::save_safetensors;

    /**
     * @brief Load tensors from a .safetensors file.
     */
    using ::safetensors::load;

    /**
     * @brief Save a map of named tensors to a .safetensors file.
     */
    using ::safetensors::save;

    /**
     * @brief Load a single tensor from a .safetensors file with only one tensor.
     */
    using ::safetensors::load_safetensor;

    /**
     * @brief Save a single tensor to a .safetensors file.
     */
    using ::safetensors::save_safetensor;

} // yann::utils

#endif //YANN_SAFETENSORS_HPP
