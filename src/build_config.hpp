#pragma once

#include <cstdint>
#include <cfloat>
#include <stdfloat>

/* Define single precision (32-bit floating point) if any precision is not defined */
#if !defined(_USE_DOUBLE_PRECISION) && !defined(_USE_SINGLE_PRECISION) && !defined(_USE_HALF_PRECISION) && !defined(_USE_QUARTER_PRECISION)
    #define _USE_SINGLE_PRECISION
#endif

/* Define _USE_EIGEN if any API is not defined */
#if !defined(_USE_EIGEN) && !defined(_USE_ONEAPI) && !defined(_USE_CUDA) && !defined(_USE_NATIVE_CPU)
    #define _USE_EIGEN
#endif

#include "math_api.hpp"
namespace YANN
{

}