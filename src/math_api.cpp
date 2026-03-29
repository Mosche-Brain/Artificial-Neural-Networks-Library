#include "math_api.hpp"

namespace YANN::math_api 
{
    /* matrix operations */

    matrix_t createMatrix(size_t rows, size_t cols)
    {
        #ifdef _USE_EIGEN
            return matrix_t(rows, cols);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix creation here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix creation here
            return matrix_t(); // Placeholder
        #elif defined(_USE_NATIVE_CPU)
            return matrix_t(rows, std::vector<numeric_t>(cols));
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, _USE_CUDA, or _USE_NATIVE_CPU."
        #endif
    }

    matrix_t createMatrix(size_t rows, size_t cols, numeric_t initialValue)
    {
        #ifdef _USE_EIGEN
            return matrix_t::Constant(rows, cols, initialValue);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix creation with initial value here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix creation with initial value here
            return matrix_t(); // Placeholder
        #elif defined(_USE_NATIVE_CPU)
            return matrix_t(rows, std::vector<numeric_t>(cols, initialValue));
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, _USE_CUDA, or _USE_NATIVE_CPU."
        #endif
    }

    matrix_t createMatrix(std::initializer_list<std::initializer_list<numeric_t>> values)
    {
        #ifdef _USE_EIGEN
            size_t rows = values.size();
            size_t cols = values.begin()->size();
            matrix_t mat(rows, cols);
            size_t i = 0;
            for (const auto& row : values) {
                std::copy(row.begin(), row.end(), mat.row(i).data());
                ++i;
            }
            return mat;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix creation from initializer list here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix creation from initializer list here
            return matrix_t(); // Placeholder
        #elif defined(_USE_NATIVE_CPU)
            size_t rows = values.size();
            size_t cols = values.begin()->size();
            matrix_t mat(rows, std::vector<numeric_t>(cols));
            size_t i = 0;
            for (const auto& row : values) {
                std::copy(row.begin(), row.end(), mat[i].begin());
                ++i;
            }
            return mat;
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, _USE_CUDA, or _USE_NATIVE_CPU."
        #endif
    }

    matrix_t matrixAdd(const matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN
            return a + b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix addition here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix addition here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixAddInPlace(matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN
            a += b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place matrix addition here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place matrix addition here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place matrix addition here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixMultiply(const matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN
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
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixMultiplyInPlace(matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN
            a *= b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place matrix multiplication here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place matrix multiplication here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place matrix multiplication here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixScalarMultiply(const matrix_t& a, numeric_t scalar)
    {
        #ifdef _USE_EIGEN
            return a * scalar;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix-scalar multiplication here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix-scalar multiplication here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixScalarMultiplyInPlace(matrix_t& a, numeric_t scalar)
    {
        #ifdef _USE_EIGEN
            a *= scalar;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place matrix-scalar multiplication here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place matrix-scalar multiplication here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place matrix-scalar multiplication here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixElementwiseMultiply(const matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN
            return a.cwiseProduct(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI element-wise multiplication here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA element-wise multiplication here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixElementwiseMultiplyInPlace(matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN
            a = a.cwiseProduct(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place element-wise multiplication here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place element-wise multiplication here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place element-wise multiplication here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }


    matrix_t matrixSubtract(const matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN
            return a - b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix subtraction here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix subtraction here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixSubtractInPlace(matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN
            a -= b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place matrix subtraction here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place matrix subtraction here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place matrix subtraction here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixDivide(const matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN
            return a.cwiseQuotient(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix division here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix division here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixDivideInPlace(matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN
            a = a.cwiseQuotient(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place matrix division here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place matrix division here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place matrix division here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixElementwiseDivideInPlace(matrix_t& a, const matrix_t& b)
    {
        #ifdef _USE_EIGEN
            a = a.cwiseQuotient(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place element-wise division here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place element-wise division here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place element-wise division here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixScalarDivide(const matrix_t& a, numeric_t scalar)
    {
        #ifdef _USE_EIGEN
            return a / scalar;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix-scalar division here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix-scalar division here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixScalarDivideInPlace(matrix_t& a, numeric_t scalar)
    {
        #ifdef _USE_EIGEN
            a /= scalar;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place matrix-scalar division here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place matrix-scalar division here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place matrix-scalar division here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }
    
    matrix_t matrixTranspose(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.transpose();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix transpose here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix transpose here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixTransposeInPlace(matrix_t& a)
    {
        #ifdef _USE_EIGEN
            a.transposeInPlace();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place matrix transpose here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place matrix transpose here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place matrix transpose here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixTransform(const matrix_t& a, numeric_t (*func)(numeric_t))
    {
        #ifdef _USE_EIGEN
            return a.unaryExpr(func);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix transform here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix transform here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixTransformInPlace(matrix_t& a, numeric_t (*func)(numeric_t))
    {
        #ifdef _USE_EIGEN
            a = a.unaryExpr(func);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place matrix transform here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place matrix transform here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place matrix transform here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    numeric_t matrixSum(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.sum();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix sum here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix sum here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    numeric_t matrixMean(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.mean();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix mean here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix mean here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }
    
    numeric_t matrixMin(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.minCoeff();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix min here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix min here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    numeric_t matrixMax(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.maxCoeff();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix max here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix max here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    size_t matrixRows(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.rows();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix row count here
            return 0; // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix row count here
            return 0; // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    size_t matrixCols(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.cols();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix column count here
            return 0; // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix column count here
            return 0; // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixRow(const matrix_t& a, size_t row)
    {
        #ifdef _USE_EIGEN
            return a.row(row);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix row access here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix row access here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixCol(const matrix_t& a, size_t col)
    {
        #ifdef _USE_EIGEN
            return a.col(col);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix column access here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix column access here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    numeric_t matrixAt(const matrix_t& a, size_t row, size_t col)
    {
        #ifdef _USE_EIGEN
            return a(row, col);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix cell access here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix cell access here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    /* vector operations */

    vector_t vectorAdd(const vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            return a + b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector addition here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector addition here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void vectorAddInPlace(vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            a += b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place vector addition here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place vector addition here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place vector addition here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }


    vector_t vectorScalarMultiply(const vector_t& a, numeric_t scalar)
    {
        #ifdef _USE_EIGEN
            return a * scalar;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector-scalar multiplication here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector-scalar multiplication here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void vectorScalarMultiplyInPlace(vector_t& a, numeric_t scalar)
    {
        #ifdef _USE_EIGEN
            a *= scalar;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place vector-scalar multiplication here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place vector-scalar multiplication here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place vector-scalar multiplication here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    vector_t vectorMultiply(const vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            return a.cwiseProduct(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector multiplication here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector multiplication here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void vectorMultiplyInPlace(vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            a = a.cwiseProduct(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place vector multiplication here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place vector multiplication here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place vector multiplication here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    numeric_t vectorDot(const vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            return a.dot(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector dot product here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector dot product here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void vectorDotInPlace(numeric_t& result, const vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            result = a.dot(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place vector dot product here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place vector dot product here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place vector dot product here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    vector_t vectorSubtract(const vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            return a - b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector subtraction here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector subtraction here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void vectorSubtractInPlace(vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            a -= b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place vector subtraction here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place vector subtraction here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place vector subtraction here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    vector_t vectorDivide(const vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            return a.cwiseQuotient(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector division here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector division here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void vectorDivideInPlace(vector_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            a = a.cwiseQuotient(b);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place vector division here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place vector division here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place vector division here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    vector_t vectorScalarDivide(const vector_t& a, numeric_t scalar)
    {
        #ifdef _USE_EIGEN
            return a / scalar;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector-scalar division here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector-scalar division here
            return vector_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void vectorScalarDivideInPlace(vector_t& a, numeric_t scalar)
    {
        #ifdef _USE_EIGEN
            a /= scalar;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place vector-scalar division here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place vector-scalar division here
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place vector-scalar division here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    numeric_t vectorSum(const vector_t& a)
    {
        #ifdef _USE_EIGEN
            return a.sum();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector sum here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector sum here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    size_t vectorSize(const vector_t& a)
    {
        #ifdef _USE_EIGEN
            return a.size();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector size here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector size here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    numeric_t vectorAt(const vector_t& a, size_t index)
    {
        #ifdef _USE_EIGEN
            return a[index];
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector element access here
            return numeric_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector element access here
            return numeric_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }
}