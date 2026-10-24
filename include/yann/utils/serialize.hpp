/**
 * @file serialize.hpp
 * @author jaro
 * @brief Model serialization and deserialization routines to and from safetensors format.
 * @date 2026-09-29
 */

#pragma once

#include "safetensors.hpp"
#include "Sequential.hpp"

namespace yann::models { class Sequential; }

namespace yann::utils
{
    /**
     * @brief Serialize a Sequential model's layer parameters into a safetensors file.
     * @param model Reference to the Sequential model to serialize.
     * @param filename Path to destination .safetensors file.
     */
    void model_serialize_to_safetensors(const models::Sequential& model, const char* filename);

    /**
     * @brief Deserialize and restore model parameters from a safetensors file into a Sequential model.
     * @param model Reference to the Sequential model to populate.
     * @param filename Path to source .safetensors file.
     */
    void model_deserialize_from_safetensors(models::Sequential& model, const char* filename);
    // models::Sequential model_deserialize_from_safetensors(const char* filename);
}
