#include "math_api.hpp"

namespace YANN::math_api 
{
    void setUsedThreadCount(size_t threadCount)
    {
        #ifdef _USE_EIGEN
            Eigen::setNbThreads(threadCount);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI thread count setting here
        #elif defined(_USE_CUDA)
            // Implement CUDA thread count setting here
        #elif defined(_USE_OPENBLAS)
            // placeholder
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU thread count setting here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, _USE_CUDA, or _USE_NATIVE_CPU."
        #endif
    }

    size_t getUsedThreadCount()
    {
        #ifdef _USE_EIGEN
            return Eigen::nbThreads();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI thread count retrieval here
            return 0; // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA thread count retrieval here
            return 0; // Placeholder
        #elif defined(_USE_OPENBLAS)
            // placeholder
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU thread count retrieval here
            return 0; // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, _USE_CUDA, or _USE_NATIVE_CPU."
        #endif
    }

    size_t getDeviceThreadCount() 
    {
        #ifdef _USE_EIGEN
            return Eigen::nbThreads();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI device thread count retrieval here
            return 0; // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA device thread count retrieval here
            return 0; // Placeholder
        #elif defined(_USE_OPENBLAS)
            // placeholder
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU device thread count retrieval here
            return 0; // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, _USE_CUDA, or _USE_NATIVE_CPU."
        #endif
    }

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
        #elif defined(_USE_OPENBLAS)
            matrix_t mat;
            mat.rows = rows;
            mat.cols = cols;
            mat.data = new double[rows * cols]();
            return mat;
    
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
        #elif defined(_USE_OPENBLAS)
            matrix_t mat;
            mat.rows = rows;
            mat.cols = cols;
            mat.data = new numeric_t[rows * cols];

            #pragma omp parallel for
            for(size_t i = 0; i < rows * cols; ++i) 
            // std::fill(mat.data, mat.data + rows * cols, initialValue);
            return mat;
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
            size_t cols = rows > 0 ? values.begin()->size() : 0;
            matrix_t mat(rows, cols);
            
            size_t i = 0;
            for (const auto& row : values) 
            {
                mat.row(i) = Eigen::Map<const vector_t>(row.begin(), row.size());
                ++i;
            }
            return mat;

        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix creation from initializer list here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix creation from initializer list here
            return matrix_t(); // Placeholder
            #elif defined(_USE_OPENBLAS)

                // placeholder
        #elif defined(_USE_NATIVE_CPU)
            size_t rows = values.size();
            size_t cols = rows > 0 ? values.begin()->size() : 0;
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

    matrix_t createRandomMatrix(size_t rows, size_t cols, numeric_t minValue, numeric_t maxValue)
    {
        #ifdef _USE_EIGEN
            return matrix_t::Random(rows, cols);
            // return matrix_t::Random(rows, cols) * ((maxValue - minValue) / 2 + (minValue + maxValue) / 2);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI random matrix creation here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA random matrix creation here
            return matrix_t(); // Placeholder
        #elif defined(_USE_OPENBLAS)
            // placeholder
        #elif defined(_USE_NATIVE_CPU)
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            #ifdef _USE_DOUBLE_PRECISION
                return openblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a.rows, b.cols, a.cols, 1.0, a.data, a.cols, b.data, b.cols, 0.0, c.data, c.cols);
            #elif defined(_USE_SINGLE_PRECISION)
                return openblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a.rows, b.cols, a.cols, 1.0, a.data, a.cols, b.data, b.cols, 0.0, c.data, c.cols);
            #elif defined(_USE_HALF_PRECISION)
                return openblas_shgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, a.rows, b.cols, a.cols, 1.0f, a.data, a.cols, b.data, b.cols, 0.0f, c.data, c.cols);
            #else
                #error "No precision defined. Please define one of _USE_DOUBLE_PRECISION or _USE_SINGLE_PRECISION."
            #endif
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place matrix transform here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixNormalize(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return (a.rowwise() - a.colwise().minCoeff()).array().rowwise() / (a.colwise().maxCoeff() - a.colwise().minCoeff()).array();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix normalization here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix normalization here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixNormalizeInPlace(matrix_t& a)
    {
        #ifdef _USE_EIGEN
            a = (a.rowwise() - a.colwise().minCoeff()).array().rowwise() / (a.colwise().maxCoeff() - a.colwise().minCoeff()).array();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place matrix normalization here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place matrix normalization here
        #elif defined(_USE_OPENBLAS)
            // placeholder
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place matrix normalization here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixColwise(const matrix_t& a, numeric_t (*func)(const vector_t&))
    {
        #ifdef _USE_EIGEN
            // implement column-wise operation using Eigen's colwise() and unaryExpr()
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI column-wise operation here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA column-wise operation here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixRowwise(const matrix_t& a, numeric_t (*func)(const vector_t&))
    {
        #ifdef _USE_EIGEN
            // implement row-wise operation using Eigen's rowwise() and unaryExpr()
            return matrix_t(); // Placeholder
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI row-wise operation here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA row-wise operation here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixColwiseSum(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.colwise().sum();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI column-wise sum here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA column-wise sum here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixRowwiseSum(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.rowwise().sum();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI row-wise sum here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA row-wise sum here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixColwiseMean(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.colwise().mean();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI column-wise mean here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA column-wise mean here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixRowwiseMean(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.rowwise().mean();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI row-wise mean here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA row-wise mean here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixColwiseMin(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.colwise().minCoeff();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI column-wise min here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA column-wise min here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixRowwiseMin(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.rowwise().minCoeff();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI row-wise min here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA row-wise min here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixColwiseMax(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.colwise().maxCoeff();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI column-wise max here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA column-wise max here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixRowwiseMax(const matrix_t& a)
    {
        #ifdef _USE_EIGEN
            return a.rowwise().maxCoeff();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI row-wise max here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA row-wise max here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixColwiseAdd(const matrix_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            return a.colwise() + b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI column-wise addition here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA column-wise addition here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixColwiseAddInPlace(matrix_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            a.colwise() += b;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place column-wise addition here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place column-wise addition here
        #elif defined(_USE_OPENBLAS)
            // placeholder
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place column-wise addition here
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    matrix_t matrixRowwiseAdd(const matrix_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            return a.rowwise() + b.transpose();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI row-wise addition here
            return matrix_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA row-wise addition here
            return matrix_t(); // Placeholder
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixRowwiseAddInPlace(matrix_t& a, const vector_t& b)
    {
        #ifdef _USE_EIGEN
            a.rowwise() += b.transpose();
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI in-place row-wise addition here
        #elif defined(_USE_CUDA)
            // Implement CUDA in-place row-wise addition here
        #elif defined(_USE_OPENBLAS)
            // placeholder
        #elif defined(_USE_NATIVE_CPU)
            // Implement native CPU in-place row-wise addition here
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
        #elif defined(_USE_OPENBLAS) // row-major order
            matrix_t rowVec(1, matrixCols(a));
            for (size_t j = 0; j < matrixCols(a); ++j) {
                rowVec(0, j) = a(row, j);
            }
            return rowVec;
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
        #elif defined(_USE_OPENBLAS) // row-major order
            matrix_t colVec(matrixRows(a), 1);
            for (size_t i = 0; i < matrixRows(a); ++i) {
                colVec(i, 0) = a(i, col);
            }   
            return colVec;
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
        #elif defined(_USE_OPENBLAS)
            return a[row * matrixCols(a) + col]; // Assuming row-major order
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixSetRow(matrix_t& a, size_t row, const vector_t& values)
    {
        #ifdef _USE_EIGEN
            a.row(row) = values;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix row setting here
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix row setting here
        #elif defined(_USE_OPENBLAS)
            for (size_t j = 0; j < matrixCols(a); ++j) {
                a[row * matrixCols(a) + j] = values(j);
            }
        #elif defined(_USE_NATIVE_CPU)
            std::copy(values.begin(), values.end(), a[row].begin());
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixSetCol(matrix_t& a, size_t col, const vector_t& values)
    {
        #ifdef _USE_EIGEN
            a.col(col) = values;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix column setting here
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix column setting here
        #elif defined(_USE_OPENBLAS)
            // placeholder
        #elif defined(_USE_NATIVE_CPU)
            for (size_t i = 0; i < a.size(); ++i) {
                a[i][col] = values[i];
            }
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    void matrixSetAt(matrix_t& a, size_t row, size_t col, numeric_t value)
    {
        #ifdef _USE_EIGEN
            a(row, col) = value;
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI matrix cell setting here
        #elif defined(_USE_CUDA)
            // Implement CUDA matrix cell setting here
        #elif defined(_USE_OPENBLAS)
            a[row * matrixCols(a) + col] = value; // Assuming row-major order
        #elif defined(_USE_NATIVE_CPU)
            a[row][col] = value;
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, or _USE_CUDA."
        #endif
    }

    /* vector operations */

    vector_t createVector(size_t size, numeric_t initialValue)
    {
        #ifdef _USE_EIGEN
            return vector_t::Constant(size, initialValue);
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector creation with initial value here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector creation with initial value here
            return vector_t(); // Placeholder
        #elif defined(_USE_OPENBLAS)
            // placeholder
        #elif defined(_USE_NATIVE_CPU)
            return vector_t(size, initialValue);
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, _USE_CUDA, or _USE_NATIVE_CPU."
        #endif
    }

    vector_t createVector(std::initializer_list<numeric_t> values)
    {
        #ifdef _USE_EIGEN
            return Eigen::Map<const vector_t>(values.begin(), values.size());
        #elif defined(_USE_ONEAPI)
            // Implement oneAPI vector creation from initializer list here
            return vector_t(); // Placeholder
        #elif defined(_USE_CUDA)
            // Implement CUDA vector creation from initializer list here
            return vector_t(); // Placeholder
        #elif defined(_USE_OPENBLAS)
            // placeholder
        #elif defined(_USE_NATIVE_CPU)
            return vector_t(values);
        #else
            #error "No math API defined. Please define one of _USE_EIGEN, _USE_ONEAPI, _USE_CUDA, or _USE_NATIVE_CPU."
        #endif
    }

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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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
        #elif defined(_USE_OPENBLAS)
            // placeholder
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