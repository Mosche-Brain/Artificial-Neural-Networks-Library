//
// Created by jaro on 7/19/26.
//

#pragma once

namespace cum::functions
{
    enum class function_id : std::uint8_t
    {
        undefined = 0,
        linear,
        relu,
        leaky_relu,
        gelu,
        elu,
        exp, // base e
        exp2,
        exp10,
        log, // ln (base e)
        log2,
        log10,
        sin,
        cos,
        tan,
        sinh,
        cosh,
        tanh,
        asin,
        acos,
        atan,
        sigmoid
    };

    typedef cumeric_t (*function_t)(cumeric_t);

    typedef struct
    {
        function_id name;
        function_t function;
        function_t derivative;
    } activation_t;
}


