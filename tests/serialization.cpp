/*
 * @author: jaro
 * @name:   serialization
 * @file:   tests/serialization.cpp
 * @date:   29 September 2026 15:51:08
 */

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <print>
#include <ranges>
#include <string>

#include <cum/cum.hpp>

#include "yann/utils/serialize.hpp"
#include "yann/models/Sequential.hpp"
#include "yann/models/layers/Dense.hpp"
#include "yann/models/layers/Input.hpp"

using namespace yann::utils;
using namespace yann::models;

TEST_CASE("serializing and deserializing sequential model")
{
    cum::cum(cum::DEVICE::CPU);

    Sequential model_a({
        layers::Input::createUnique(27),
        layers::Dense::createUnique(256, "relu"),
        layers::Dense::createUnique(128, "relu"),
        layers::Dense::createUnique(64, "relu"),
        layers::Dense::createUnique(1, "sigmoid")
    });

    Sequential model_b({
        layers::Input::createUnique(27),
        layers::Dense::createUnique(256, "relu"),
        layers::Dense::createUnique(128, "relu"),
        layers::Dense::createUnique(64, "relu"),
        layers::Dense::createUnique(1, "sigmoid")
    });

    model_serialize_to_safetensors(model_a, "model_a.safetensors");
    model_deserialize_from_safetensors(model_b, "model_a.safetensors");

    for (const auto [layer_a, layer_b] : std::views::zip(model_a.getTopology(), model_b.getTopology()))
    {
        REQUIRE(layer_a->weights() == layer_b->weights());
        REQUIRE(layer_a->biases() == layer_b->biases());
    }
}