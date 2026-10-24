/**
 * @file build_config.hpp
 * @brief Compile-time configuration flags and precision definitions for the YANN framework.
 *
 * This header defines default floating-point precision levels and underlying
 * mathematical backend defaults if none were explicitly configured during build.
 */

#pragma once

#include <cstdint>
#include <cfloat>
#include <stdfloat>

/**
 * @def _USE_SINGLE_PRECISION
 * @brief Default 32-bit floating point precision flag if no specific precision is specified.
 */
/* Define single precision (32-bit floating point) if any precision is not defined */
#if !defined(_USE_DOUBLE_PRECISION) && !defined(_USE_SINGLE_PRECISION) && !defined(_USE_HALF_PRECISION) && !defined(_USE_QUARTER_PRECISION)
    #define _USE_SINGLE_PRECISION
#endif

/**
 * @def _USE_EIGEN
 * @brief Default backend API selection flag if no compute backend is explicitly enabled.
 */
/* Define _USE_EIGEN if any API is not defined */
#if !defined(_USE_EIGEN) && !defined(_USE_ONEAPI) && !defined(_USE_CUDA) && !defined(_USE_NATIVE_CPU)
    #define _USE_EIGEN
#endif

// #include "math_api.hpp"
