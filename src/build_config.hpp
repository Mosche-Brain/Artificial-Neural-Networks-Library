#pragma once

#include <cstdint>
#include <cfloat>

/* Define used precision */
// #define _USE_FULL_PRECISION
#define _USE_HALF_PRECISION
// #define _USE_QUARTER_PRECISION

/* Define used math API */
#define _USE_EIGEN_CPU
// #define _USE_ONEAPI
// #define _USE_CUDA
// #define _USE_NATIVE_CPU

#ifdef _USE_EIGEN_CPU
#include <eigen3/Eigen/Dense>
#endif

namespace YANN
{
    #ifdef _USE_FULL_PRECISION
        typedef float numeric_t;
    #elif defined(_USE_HALF_PRECISION)
        typedef _Float16 numeric_t;
    #elif defined(_USE_QUARTER_PRECISION)
        typedef int8_t numeric_t;
    #else
        #error "No precision defined. Please define one of _USE_FULL_PRECISION, _USE_HALF_PRECISION, or _USE_QUARTER_PRECISION."
    #endif
    
    #ifdef _USE_EIGEN_CPU
        typedef Eigen::Matrix<numeric_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
        typedef Eigen::Matrix<numeric_t, Eigen::Dynamic, 1> vector_t;
    #elif defined(_USE_ONEAPI)
        // Define matrix_t and vector_t for oneAPI here
    #elif defined(_USE_CUDA)
        // Define matrix_t and vector_t for CUDA here
    #elif defined(_USE_NATIVE_CPU)
        // Define matrix_t and vector_t for native CPU here
    #endif
}