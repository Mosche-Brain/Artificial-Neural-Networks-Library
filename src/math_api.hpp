#pragma once

#include "build_config.hpp"

namespace YANN::math_api
{
    matrix_t matrixMultiply(const matrix_t& a, const matrix_t& b);
    matrix_t matrixAdd(const matrix_t& a, const matrix_t& b);
    matrix_t matrixSubtract(const matrix_t& a, const matrix_t& b);
    matrix_t matrixTranspose(const matrix_t& a);
    matrix_t matrixScalarMultiply(const matrix_t& a, numeric_t scalar);
    matrix_t matrixElementwiseMultiply(const matrix_t& a, const matrix_t& b);
    matrix_t matrixRow(const matrix_t& a, int row);
    matrix_t matrixCol(const matrix_t& a, int col);
    numeric_t matrixCell(const matrix_t& a, int row, int col);
    
    vector_t vectorAdd(const vector_t& a, const vector_t& b);
    vector_t vectorSubtract(const vector_t& a, const vector_t& b);
    vector_t vectorScalarMultiply(const vector_t& a, numeric_t scalar);
    vector_t vectorMultiply(const vector_t& a, const vector_t& b);
    numeric_t vectorDot(const vector_t& a, const vector_t& b);
    vector_t vectorAt(const vector_t& a, int index);


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
        typedef std::vector<std::vector<numeric_t>> matrix_t;
        typedef std::vector<numeric_t> vector_t;
    #else
        #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
    #endif
}