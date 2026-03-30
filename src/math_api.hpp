#pragma once

#include "build_config.hpp"

#include <stdfloat>

#ifdef _USE_EIGEN
    #include <eigen3/Eigen/Dense>
    #include <eigen3/unsupported/Eigen/CXX11/Tensor>
#elif defined(_USE_ONEAPI)
    #include <oneapi/dnnl.hpp>
#elif defined(_USE_CUDA)
    #include <cuda_runtime.h>
#elif defined(_USE_NATIVE_CPU)
    #include <vector>
#else
    #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, _USE_CUDA, or _USE_NATIVE_CPU."
#endif
namespace YANN::math_api
{
    /* numeric type definition */
    #ifdef _USE_DOUBLE_PRECISION
        typedef double numeric_t;
    #elif defined(_USE_SINGLE_PRECISION)
        typedef float numeric_t;
    #elif defined(_USE_HALF_PRECISION)
        typedef std::float16_t numeric_t;
    #elif defined(_USE_BRAIN_PRECISION)
        typedef std::bfloat16_t numeric_t;
    #elif defined(_USE_QUARTER_PRECISION)
        typedef int8_t numeric_t;
    #else
        #error "No precision defined. Please define one of  _USE_DOUBLE_PRECISION, _USE_SINGLE_PRECISION, _USE_HALF_PRECISION, or _USE_QUARTER_PRECISION."
    #endif

    /* tensor types definitiona */

    #ifdef _USE_EIGEN
        // #include <eigen3/Eigen/Dense>
        // #include <eigen3/unsupported/Eigen/CXX11/Tensor>

        template<size_t n>
        using tensor_t = Eigen::Tensor<numeric_t, n>;

        using matrix_t = Eigen::Matrix<numeric_t, Eigen::Dynamic, Eigen::Dynamic>;
        using vector_t = Eigen::Matrix<numeric_t, Eigen::Dynamic, 1>;
        using scalar_t = numeric_t;
    #elif defined(_USE_ONEAPI)
        #error "OneAPI backend not implemented yet. Plese define _USE_EIGEN_CPU instead."
    #elif defined(_USE_CUDA)
        #error "CUDA backend not implemented yet. Plese define _USE_EIGEN_CPU instead."
    #elif defined(_USE_NATIVE_CPU)
        using matrix_t = std::vector<std::vector<numeric_t>>;
        using vector_t = std::vector<numeric_t>;
        using scalar_t = numeric_t;
    #else
        #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
    #endif

    /* matrix operations */

    matrix_t createMatrix(size_t rows, size_t cols);
    matrix_t createMatrix(size_t rows, size_t cols, numeric_t initialValue);
    matrix_t createMatrix(std::initializer_list<std::initializer_list<numeric_t>> values);

    matrix_t createRandomMatrix(size_t rows, size_t cols, numeric_t minValue = static_cast<numeric_t>(-1), numeric_t maxValue = static_cast<numeric_t>(1));

    matrix_t matrixAdd(const matrix_t& a, const matrix_t& b);
    void matrixAddInPlace(matrix_t& a, const matrix_t& b);

    matrix_t matrixMultiply(const matrix_t& a, const matrix_t& b);
    void matrixMultiplyInPlace(matrix_t& a, const matrix_t& b);

    matrix_t matrixElementwiseMultiply(const matrix_t& a, const matrix_t& b);
    void matrixElementwiseMultiplyInPlace(matrix_t& a, const matrix_t& b);

    matrix_t matrixScalarMultiply(const matrix_t& a, numeric_t scalar);
    void matrixScalarMultiplyInPlace(matrix_t& a, numeric_t scalar);

    matrix_t matrixSubtract(const matrix_t& a, const matrix_t& b);
    void matrixSubtractInPlace(matrix_t& a, const matrix_t& b);

    matrix_t matrixDivide(const matrix_t& a, const matrix_t& b);
    void matrixDivideInPlace(matrix_t& a, const matrix_t& b);

    matrix_t matrixElementwiseDivide(const matrix_t& a, const matrix_t& b);
    void matrixElementwiseDivideInPlace(matrix_t& a, const matrix_t& b);

    matrix_t matrixScalarDivide(const matrix_t& a, numeric_t scalar);
    void matrixScalarDivideInPlace(matrix_t& a, numeric_t scalar);

    matrix_t matrixTranspose(const matrix_t& a);
    void matrixTransposeInPlace(matrix_t& a);

    matrix_t matrixTransform(const matrix_t& a, numeric_t (*func)(numeric_t));
    void matrixTransformInPlace(matrix_t& a, numeric_t (*func)(numeric_t));

    // add matrixColwise and matrixRowwise operations now <------- do it
    matrix_t matrixColwise(const matrix_t& a, numeric_t (*func)(const vector_t&));
    matrix_t matrixRowwise(const matrix_t& a, numeric_t (*func)(const vector_t&));

    matrix_t matrixColwiseSum(const matrix_t& a);
    matrix_t matrixRowwiseSum(const matrix_t& a);

    matrix_t matrixColwiseMean(const matrix_t& a);
    matrix_t matrixRowwiseMean(const matrix_t& a);

    matrix_t matrixColwiseMin(const matrix_t& a);
    matrix_t matrixRowwiseMin(const matrix_t& a);

    matrix_t matrixColwiseMax(const matrix_t& a);
    matrix_t matrixRowwiseMax(const matrix_t& a);

    matrix_t matrixColwiseAdd(const matrix_t& a, const vector_t& b);
    void matrixColwiseAddInPlace(matrix_t& a, const vector_t& b);

    matrix_t matrixRowwiseAdd(const matrix_t& a, const vector_t& b);
    void matrixRowwiseAddInPlace(matrix_t& a, const vector_t& b);

    numeric_t matrixSum(const matrix_t& a);
    numeric_t matrixMean(const matrix_t& a);
    numeric_t matrixMin(const matrix_t& a);
    numeric_t matrixMax(const matrix_t& a);

    size_t matrixRows(const matrix_t& a);
    size_t matrixCols(const matrix_t& a);
            
    matrix_t  matrixRow(const matrix_t& a, size_t row);
    matrix_t  matrixCol(const matrix_t& a, size_t col);
    numeric_t matrixAt(const matrix_t& a, size_t row, size_t col);
    
    /* vector operations */

    vector_t createVector(size_t size);
    vector_t createVector(size_t size, numeric_t initialValue);
    vector_t createVector(std::initializer_list<numeric_t> values);

    vector_t vectorAdd(const vector_t& a, const vector_t& b);
    void vectorAddInPlace(vector_t& a, const vector_t& b);
    
    vector_t vectorMultiply(const vector_t& a, const vector_t& b);
    void vectorMultiplyInPlace(vector_t& a, const vector_t& b);

    vector_t vectorScalarMultiply(const vector_t& a, numeric_t scalar);
    void vectorScalarMultiplyInPlace(vector_t& a, numeric_t scalar);

    numeric_t vectorDot(const vector_t& a, const vector_t& b);
    void vectorDotInPlace(numeric_t& result, const vector_t& a, const vector_t& b);

    vector_t vectorSubtract(const vector_t& a, const vector_t& b);
    void vectorSubtractInPlace(vector_t& a, const vector_t& b);

    vector_t vectorDivide(const vector_t& a, const vector_t& b);
    void vectorDivideInPlace(vector_t& a, const vector_t& b);

    vector_t vectorElementwiseDivide(const vector_t& a, const vector_t& b);
    void vectorElementwiseDivideInPlace(vector_t& a, const vector_t& b);

    vector_t vectorScalarDivide(const vector_t& a, numeric_t scalar);
    void vectorScalarDivideInPlace(vector_t& a, numeric_t scalar);

    vector_t vectorTransform(const vector_t& a, numeric_t (*func)(numeric_t));
    void vectorTransformInPlace(vector_t& a, numeric_t (*func)(numeric_t));

    numeric_t vectorSum(const vector_t& a);

    size_t vectorSize(const vector_t& a);

    numeric_t vectorAt(const vector_t& a, size_t index);

}
namespace YANN
{
    using numeric_t = YANN::math_api::numeric_t;
    using matrix_t = YANN::math_api::matrix_t;
    using vector_t = YANN::math_api::vector_t;
}