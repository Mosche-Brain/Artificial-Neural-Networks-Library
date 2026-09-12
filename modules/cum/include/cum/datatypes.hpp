/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Świadomy Mężczyzna
 * @email       : 
 * @file        : modules/cum/include/cum/datatypes.hpp
 * @createdOn   : 09/12/26
 * @description : Datatypes definitions for CUM
 *---------------------------------------------**/

#pragma once

#include "cum/config.hpp"

#include <stdfloat>
#include <cstdint>

#if CUM_USE_MKL
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#endif

namespace cum
{
    using float64 = double;
    using float32 = float;
#if CUM_USE_MKL
    using float16 = sycl::half;
    using bfloat16 = sycl::ext::oneapi::bfloat16;
#else
#ifdef __STDCPP_FLOAT16_T__
    using float16 = std::float16;
#else
    using float16 = float;
#warning float16 not supported by compiler, fallback to float32
#endif
#ifdef __STDCPP_BFLOAT16_T__
    using bfloat16 = std::bfloat16;
#else
    using bfloat16 = float;
#warning bfloat16 not supported by compiler, fallback to float32
#endif
#endif
    typedef struct fp8_e4_3m_impl
    {
        unsigned char data;
    } fp8_e4_3m_impl;
    using float8 = fp8_e4_3m_impl;

    using int64 = signed long long;
    using int32 = signed int;
    using int16 = signed short;
    using int8  = signed char;
    
    using uint64 = unsigned long long;
    using uint32 = unsigned int;
    using uint16 = unsigned short;
    using uint8  = unsigned char;

    using f64   = float64;
    using f32   = float32;
    using f16   = float16;
    using bf16  = bfloat16;
    using f8    = float8;

    using i64   = int64;
    using i32   = int32;
    using i16   = int16;
    using i8    = int8;

    using u64   = uint64;
    using u32   = uint32;
    using u16   = uint16;
    using u8    = uint8;

    using byte = uint8;
}