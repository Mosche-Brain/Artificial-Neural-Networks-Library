/*
 * @author: jaro
 * @name:   serialize
 * @file:   include/yann/utils/serialize.hpp
 * @date:   29 September 2026 09:32:24
 */

#pragma once

#include "safetensors.hpp"
#include "Sequential.hpp"

namespace yann::models{ class Sequential; }

namespace yann::utils
{
    void model_serialize_to_safetensors(const models::Sequential& model, const char* filename);

    void model_deserialize_from_safetensors(models::Sequential& model, const char* filename);
    // models::Sequential model_deserialize_from_safetensors(const char* filename);
}