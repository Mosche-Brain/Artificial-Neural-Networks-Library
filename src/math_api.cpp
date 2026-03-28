#include "math_api.hpp"

namespace YANN::math_api{
    matrix_t matrixMultiply(const matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN_CPU
            return a * b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix multiplication here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix multiplication here
            return matrix_t(); // Placeholder
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU matrix multiplication here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixAdd(const matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN_CPU
            return a + b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix addition here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix addition here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixSubtract(const matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN_CPU
            return a - b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix subtraction here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix subtraction here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }
    
    matrix_t matrixTranspose(const matrix_t& a)
    {
        #ifdef _USE_EIGEN_CPU
            return a.transpose();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix transpose here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix transpose here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixScalarMultiply(const matrix_t& a, numeric_t scalar)
    {
        #ifdef _USE_EIGEN_CPU
            return a * scalar;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix-scalar multiplication here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix-scalar multiplication here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixElementwiseMultiply(const matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN_CPU
            return a.cwiseProduct(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI element-wise multiplication here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA element-wise multiplication here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixRow(const matrix_t& a, int row)
    {
        #ifdef _USE_EIGEN_CPU
            return a.row(row);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix row access here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix row access here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixCol(const matrix_t& a, int col)
    {
        #ifdef _USE_EIGEN_CPU
            return a.col(col);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix column access here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix column access here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    numeric_t matrixCell(const matrix_t& a, int row, int col)
    {
        #ifdef _USE_EIGEN_CPU
            return a(row, col);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix cell access here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix cell access here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    vector_t vectorAdd(const vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN_CPU
            return a + b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector addition here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector addition here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    vector_t vectorSubtract(const vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN_CPU
            return a - b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector subtraction here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector subtraction here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    vector_t vectorScalarMultiply(const vector_t& a, numeric_t scalar)
    {
        #ifdef _USE_EIGEN_CPU
            return a * scalar;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector-scalar multiplication here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector-scalar multiplication here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    vector_t vectorMultiply(const vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN_CPU
            return a.cwiseProduct(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector multiplication here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector multiplication here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    numeric_t vectorDot(const vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN_CPU
            return a.dot(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector dot product here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector dot product here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    vector_t vectorAt(const vector_t& a, int index)
    {
        #ifdef _USE_EIGEN_CPU
            return a[index];
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector element access here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector element access here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN_CPU, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }
}