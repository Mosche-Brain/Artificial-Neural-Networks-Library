#pragma once

#include "build_config.hpp"

#include <stdfloat>

namespace YANN::math_api
{
    #ifdef _USE_DOUBLE_PRECISION
        typedef double numeric_t;
    #elif defined(_USE_SINGLE_PRECISION)
        typedef float numeric_t;
    #elif defined(_USE_HALF_PRECISION)
        typedef std::float16_t numeric_t;
    #elif defined(_USE_QUARTER_PRECISION)
        typedef int8_t numeric_t;
    #else
        #error "No precision defined. Please define one of  _USE_DOUBLE_PRECISION, _USE_SINGLE_PRECISION, _USE_HALF_PRECISION, or _USE_QUARTER_PRECISION."
    #endif

    #ifdef _USE_EIGEN
        #include <eigen3/Eigen/Dense>
        using matrix_t = Eigen::Matrix<numeric_t, Eigen::Dynamic, Eigen::Dynamic>;
        using vector_t = Eigen::Matrix<numeric_t, Eigen::Dynamic, 1>;
    #elif defined(_USE_ONEAPI)
        #error "OneAPI backend not implemented yet. Plese define _USE_EIGEN_CPU instead."
    #elif defined(_USE_CUDA)
        #error "CUDA backend not implemented yet. Plese define _USE_EIGEN_CPU instead."
    #elif defined(_USE_NATIVE_CPU)
        using matrix_t = std::vector<std::vector<numeric_t>>;
        using vector_t = std::vector<numeric_t>;
    #else
        #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
    #endif


    matrix_t matrixMultiply(const matrix_t& a, const matrix_t& b);
    void matrixMultiplyInPlace(matrix_t& a, const matrix_t& b);
    matrix_t matrixAdd(const matrix_t& a, const matrix_t& b);
    void matrixAddInPlace(matrix_t& a, const matrix_t& b);
    matrix_t matrixSubtract(const matrix_t& a, const matrix_t& b);
    void matrixSubtractInPlace(matrix_t& a, const matrix_t& b);
    matrix_t matrixTranspose(const matrix_t& a);
    void matrixTransposeInPlace(matrix_t& a);
    matrix_t matrixScalarMultiply(const matrix_t& a, numeric_t scalar);
    void matrixScalarMultiplyInPlace(matrix_t& a, numeric_t scalar);
    matrix_t matrixElementwiseMultiply(const matrix_t& a, const matrix_t& b);
    void matrixElementwiseMultiplyInPlace(matrix_t& a, const matrix_t& b);
    matrix_t matrixTransform(const matrix_t& a, numeric_t (*func)(numeric_t));
    void matrixTransformInPlace(matrix_t& a, numeric_t (*func)(numeric_t));
    matrix_t matrixRow(const matrix_t& a, int row);
    matrix_t matrixCol(const matrix_t& a, int col);
    numeric_t matrixCell(const matrix_t& a, int row, int col);
    

    vector_t vectorAdd(const vector_t& a, const vector_t& b);
    void vectorAddInPlace(vector_t& a, const vector_t& b);
    vector_t vectorSubtract(const vector_t& a, const vector_t& b);
    void vectorSubtractInPlace(vector_t& a, const vector_t& b);
    vector_t vectorScalarMultiply(const vector_t& a, numeric_t scalar);
    void vectorScalarMultiplyInPlace(vector_t& a, numeric_t scalar);
    vector_t vectorMultiply(const vector_t& a, const vector_t& b);
    void vectorMultiplyInPlace(vector_t& a, const vector_t& b);
    vector_t vectorTransform(const vector_t& a, numeric_t (*func)(numeric_t));
    void vectorTransformInPlace(vector_t& a, numeric_t (*func)(numeric_t));
    numeric_t vectorDot(const vector_t& a, const vector_t& b);
    void vectorDotInPlace(numeric_t& result, const vector_t& a, const vector_t& b);
    numeric_t vectorAt(const vector_t& a, int index);

}

namespace YANN
{
    using numeric_t = YANN::math_api::numeric_t;
    using matrix_t = YANN::math_api::matrix_t;
    using vector_t = YANN::math_api::vector_t;
}