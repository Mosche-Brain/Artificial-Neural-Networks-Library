#pragma once

#include "Core.hpp"

namespace cum::functions
{
    typedef cumeric_t (*function_t)(cumeric_t);
    struct ReLU
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    struct Tanh
    {
        cumeric_t operator()(cumeric_t x) const;
    };

    // return function pointer due to passed const char*
    void (*getFunctionByName(const char* name))(cumeric_t);
} // namespace cum::functions
