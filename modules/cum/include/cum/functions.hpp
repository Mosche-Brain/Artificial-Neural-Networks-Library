#pragma once

#include "Core.hpp"

namespace cum::functions
{
    struct ReLU
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    // struct Tanh
    // {
    //     cumeric_t operator()(cumeric_t x) const;
    // };
} // namespace cum::functions
