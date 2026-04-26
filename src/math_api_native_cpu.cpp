#ifdef _USE_NATIVE_CPU
#include "math_api.hpp"

namespace YANN::math_api
{
    void setUsedThreadCount(size_t threadCount)
    {
            // Implement native CPU thread count setting here
    }

    size_t getUsedThreadCount()
    {
            // Implement native CPU thread count retrieval here
            return 0; // Placeholder
    }

    size_t getDeviceThreadCount() 
    {
            // Implement native CPU device thread count retrieval here
            return 0; // Placeholder
    }

    /* matrix operations */

    matrix_t createMatrix(size_t rows, size_t cols)
    {
            return matrix_t(rows, std::vector<numeric_t>(cols));
    }

    matrix_t createMatrix(size_t rows, size_t cols, numeric_t initialValue)
    {
            return matrix_t(rows, std::vector<numeric_t>(cols, initialValue));
    }

    matrix_t createMatrix(std::initializer_list<std::initializer_list<numeric_t>> values)
    {
            size_t rows = values.size();
            size_t cols = rows > 0 ? values.begin()->size() : 0;
            matrix_t mat(rows, std::vector<numeric_t>(cols));
            size_t i = 0;
            for (const auto& row : values) {
                std::copy(row.begin(), row.end(), mat[i].begin());
                ++i;
            }
            return mat;
    }

    matrix_t createRandomMatrix(size_t rows, size_t cols, numeric_t minValue, numeric_t maxValue)
    {
            matrix_t mat(rows, std::vector<numeric_t>(cols));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<numeric_t> dis(minValue, maxValue);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    mat[i][j] = dis(gen);
                }
            }
            return mat;
    }

    matrix_t matrixAdd(const matrix_t& a, const matrix_t& b)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixAddInPlace(matrix_t& a, const matrix_t& b)
    {
            // Implement native CPU in-place matrix addition here
    }

    matrix_t matrixMultiply(const matrix_t& a, const matrix_t& b)
    {
            #ifdef _USE_DOUBLE_PRECISION
            #elif defined(_USE_SINGLE_PRECISION)
            #elif defined(_USE_HALF_PRECISION)
            #else
            #endif
            // Implement native CPU matrix multiplication here
            return matrix_t(); // Placeholder
    }

    void matrixMultiplyInPlace(matrix_t& a, const matrix_t& b)
    {
            // Implement native CPU in-place matrix multiplication here
    }

    matrix_t matrixScalarMultiply(const matrix_t& a, numeric_t scalar)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixScalarMultiplyInPlace(matrix_t& a, numeric_t scalar)
    {
            // Implement native CPU in-place matrix-scalar multiplication here
    }

    matrix_t matrixElementwiseMultiply(const matrix_t& a, const matrix_t& b)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixElementwiseMultiplyInPlace(matrix_t& a, const matrix_t& b)
    {
            // Implement native CPU in-place element-wise multiplication here
    }


    matrix_t matrixSubtract(const matrix_t& a, const matrix_t& b)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixSubtractInPlace(matrix_t& a, const matrix_t& b)
    {
            // Implement native CPU in-place matrix subtraction here
    }

    matrix_t matrixDivide(const matrix_t& a, const matrix_t& b)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixDivideInPlace(matrix_t& a, const matrix_t& b)
    {
            // Implement native CPU in-place matrix division here
    }

    void matrixElementwiseDivideInPlace(matrix_t& a, const matrix_t& b)
    {
            // Implement native CPU in-place element-wise division here
    }

    matrix_t matrixScalarDivide(const matrix_t& a, numeric_t scalar)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixScalarDivideInPlace(matrix_t& a, numeric_t scalar)
    {
            // Implement native CPU in-place matrix-scalar division here
    }
    
    matrix_t matrixTranspose(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixTransposeInPlace(matrix_t& a)
    {
            // Implement native CPU in-place matrix transpose here
    }

    matrix_t matrixTransform(const matrix_t& a, numeric_t (*func)(numeric_t))
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixTransformInPlace(matrix_t& a, numeric_t (*func)(numeric_t))
    {
            // Implement native CPU in-place matrix transform here
    }

    matrix_t matrixNormalize(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixNormalizeInPlace(matrix_t& a)
    {
            // Implement native CPU in-place matrix normalization here
    }

    matrix_t matrixColwise(const matrix_t& a, numeric_t (*func)(const vector_t&))
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixRowwise(const matrix_t& a, numeric_t (*func)(const vector_t&))
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixColwiseSum(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixRowwiseSum(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixColwiseMean(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixRowwiseMean(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixColwiseMin(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixRowwiseMin(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixColwiseMax(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixRowwiseMax(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixColwiseAdd(const matrix_t& a, const vector_t& b)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixColwiseAddInPlace(matrix_t& a, const vector_t& b)
    {
            // Implement native CPU in-place column-wise addition here
    }

    matrix_t matrixRowwiseAdd(const matrix_t& a, const vector_t& b)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixRowwiseAddInPlace(matrix_t& a, const vector_t& b)
    {
            // Implement native CPU in-place row-wise addition here
    }

    numeric_t matrixSum(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    numeric_t matrixMean(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }
    
    numeric_t matrixMin(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    numeric_t matrixMax(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    size_t matrixRows(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    size_t matrixCols(const matrix_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixRow(const matrix_t& a, size_t row)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    matrix_t matrixCol(const matrix_t& a, size_t col)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    numeric_t matrixAt(const matrix_t& a, size_t row, size_t col)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void matrixSetRow(matrix_t& a, size_t row, const vector_t& values)
    {
            std::copy(values.begin(), values.end(), a[row].begin());
    }

    void matrixSetCol(matrix_t& a, size_t col, const vector_t& values)
    {
            for (size_t i = 0; i < a.size(); ++i) {
                a[i][col] = values[i];
            }
    }

    void matrixSetAt(matrix_t& a, size_t row, size_t col, numeric_t value)
    {
            a[row][col] = value;
    }

    /* vector operations */

    vector_t createVector(size_t size, numeric_t initialValue)
    {
            return vector_t(size, initialValue);
    }

    vector_t createVector(std::initializer_list<numeric_t> values)
    {
            return vector_t(values);
    }

    vector_t vectorAdd(const vector_t& a, const vector_t& b)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void vectorAddInPlace(vector_t& a, const vector_t& b)
    {
            // Implement native CPU in-place vector addition here
    }


    vector_t vectorScalarMultiply(const vector_t& a, numeric_t scalar)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void vectorScalarMultiplyInPlace(vector_t& a, numeric_t scalar)
    {
            // Implement native CPU in-place vector-scalar multiplication here
    }

    vector_t vectorMultiply(const vector_t& a, const vector_t& b)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void vectorMultiplyInPlace(vector_t& a, const vector_t& b)
    {
            // Implement native CPU in-place vector multiplication here
    }

    numeric_t vectorDot(const vector_t& a, const vector_t& b)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void vectorDotInPlace(numeric_t& result, const vector_t& a, const vector_t& b)
    {
            // Implement native CPU in-place vector dot product here
    }

    vector_t vectorSubtract(const vector_t& a, const vector_t& b)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void vectorSubtractInPlace(vector_t& a, const vector_t& b)
    {
            // Implement native CPU in-place vector subtraction here
    }

    vector_t vectorDivide(const vector_t& a, const vector_t& b)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void vectorDivideInPlace(vector_t& a, const vector_t& b)
    {
            // Implement native CPU in-place vector division here
    }

    vector_t vectorScalarDivide(const vector_t& a, numeric_t scalar)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    void vectorScalarDivideInPlace(vector_t& a, numeric_t scalar)
    {
            // Implement native CPU in-place vector-scalar division here
    }

    numeric_t vectorSum(const vector_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    size_t vectorSize(const vector_t& a)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }

    numeric_t vectorAt(const vector_t& a, size_t index)
    {
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
    }
}
#endif
