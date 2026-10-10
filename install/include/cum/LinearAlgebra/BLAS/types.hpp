/* copied from oneapi/mkl/2026.0/include/oneapi/mkl/blas/types.hpp */
#ifndef YANN_TYPES_HPP
#define YANN_TYPES_HPP

#include <cstdint>

namespace cum::blas
{
    enum class compute_mode : std::uint64_t // copied from oneapi/mkl/2026.0/include/oneapi/mkl/blas/types.hpp
    {
        unset            = 0x0,
        float_to_bf16    = 0x1,
        float_to_bf16x2  = 0x2,
        float_to_bf16x3  = 0x4,
        float_to_tf32    = 0x10,
        complex_3m       = 0x10000,
        any              = 0xFFFFFFFF,
        standard         = 0x20000000'00000000,
        prefer_alternate = 0x40000000'00000000,
        force_alternate  = 0x80000000'00000000,
    }; // Not supported on all backends

    enum class transpose : char
    {
        nontrans = 0,
        trans = 1,
        conjtrans = 3,
        N = 0,
        T = 1,
        C = 3
    };

    enum class uplo : char
    {
        upper = 0,
        lower = 1,
        U = 0,
        L = 1
    };

    enum class diag : char
    {
        nonunit = 0,
        unit = 1,
        N = 0,
        U = 1
    };

    enum class side : char
    {
        left = 0,
        right = 1,
        L = 0,
        R = 1
    };

    enum class offset : char
    {
        row = 0,
        column = 1,
        fix = 2,
        R = 0,
        C = 1,
        F = 2
    };

    enum class index_base : char
    {
        zero = 0,
        one  = 1,
    };
}

#endif //YANN_TYPES_HPP
