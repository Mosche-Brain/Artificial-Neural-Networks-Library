#ifdef _USE_EIGEN
#include "math_api.hpp"

namespace YANN::math_api
{
    void setUsedThreadCount(size_t threadCount)
    {
            Eigen::setNbThreads(threadCount);
    }

    size_t getUsedThreadCount()
    {
            return Eigen::nbThreads();
    }

    size_t getDeviceThreadCount() 
    {
            return Eigen::nbThreads();
    }

    /* matrix operations */

    matrix_t createMatrix(size_t rows, size_t cols)
    {
            return matrix_t(rows, cols);
    }

    matrix_t createMatrix(size_t rows, size_t cols, numeric_t initialValue)
    {
            return matrix_t::Constant(rows, cols, initialValue);
    }

    matrix_t createMatrix(std::initializer_list<std::initializer_list<numeric_t>> values)
    {
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

    }

    matrix_t createRandomMatrix(size_t rows, size_t cols, numeric_t minValue, numeric_t maxValue)
    {
            return matrix_t::Random(rows, cols);
            // return matrix_t::Random(rows, cols) * ((maxValue - minValue) / 2 + (minValue + maxValue) / 2);
    }

    matrix_t matrixAdd(const matrix_t& a, const matrix_t& b)
    {
            return a + b;
    }

    void matrixAddInPlace(matrix_t& a, const matrix_t& b)
    {
            a += b;
    }

    matrix_t matrixMultiply(const matrix_t& a, const matrix_t& b)
    {
            return a * b;
    }

    void matrixMultiplyInPlace(matrix_t& a, const matrix_t& b)
    {
            a *= b;
    }

    matrix_t matrixScalarMultiply(const matrix_t& a, numeric_t scalar)
    {
            return a * scalar;
    }

    void matrixScalarMultiplyInPlace(matrix_t& a, numeric_t scalar)
    {
            a *= scalar;
    }

    matrix_t matrixElementwiseMultiply(const matrix_t& a, const matrix_t& b)
    {
            return a.cwiseProduct(b);
    }

    void matrixElementwiseMultiplyInPlace(matrix_t& a, const matrix_t& b)
    {
            a = a.cwiseProduct(b);
    }


    matrix_t matrixSubtract(const matrix_t& a, const matrix_t& b)
    {
            return a - b;
    }

    void matrixSubtractInPlace(matrix_t& a, const matrix_t& b)
    {
            a -= b;
    }

    matrix_t matrixDivide(const matrix_t& a, const matrix_t& b)
    {
            return a.cwiseQuotient(b);
    }

    void matrixDivideInPlace(matrix_t& a, const matrix_t& b)
    {
            a = a.cwiseQuotient(b);
    }

    void matrixElementwiseDivideInPlace(matrix_t& a, const matrix_t& b)
    {
            a = a.cwiseQuotient(b);
    }

    matrix_t matrixScalarDivide(const matrix_t& a, numeric_t scalar)
    {
            return a / scalar;
    }

    void matrixScalarDivideInPlace(matrix_t& a, numeric_t scalar)
    {
            a /= scalar;
    }
    
    matrix_t matrixTranspose(const matrix_t& a)
    {
            return a.transpose();
    }

    void matrixTransposeInPlace(matrix_t& a)
    {
            a.transposeInPlace();
    }

    matrix_t matrixTransform(const matrix_t& a, numeric_t (*func)(numeric_t))
    {
            return a.unaryExpr(func);
    }

    void matrixTransformInPlace(matrix_t& a, numeric_t (*func)(numeric_t))
    {
            a = a.unaryExpr(func);
    }

    matrix_t matrixNormalize(const matrix_t& a)
    {
            return (a.rowwise() - a.colwise().minCoeff()).array().rowwise() / (a.colwise().maxCoeff() - a.colwise().minCoeff()).array();
    }

    void matrixNormalizeInPlace(matrix_t& a)
    {
            a = (a.rowwise() - a.colwise().minCoeff()).array().rowwise() / (a.colwise().maxCoeff() - a.colwise().minCoeff()).array();
    }

    matrix_t matrixColwise(const matrix_t& a, numeric_t (*func)(const vector_t&))
    {
            // implement column-wise operation using Eigen's colwise() and unaryExpr()
    }

    matrix_t matrixRowwise(const matrix_t& a, numeric_t (*func)(const vector_t&))
    {
            // implement row-wise operation using Eigen's rowwise() and unaryExpr()
            return matrix_t(); // Placeholder
    }

    matrix_t matrixColwiseSum(const matrix_t& a)
    {
            return a.colwise().sum();
    }

    matrix_t matrixRowwiseSum(const matrix_t& a)
    {
            return a.rowwise().sum();
    }

    matrix_t matrixColwiseMean(const matrix_t& a)
    {
            return a.colwise().mean();
    }

    matrix_t matrixRowwiseMean(const matrix_t& a)
    {
            return a.rowwise().mean();
    }

    matrix_t matrixColwiseMin(const matrix_t& a)
    {
            return a.colwise().minCoeff();
    }

    matrix_t matrixRowwiseMin(const matrix_t& a)
    {
            return a.rowwise().minCoeff();
    }

    matrix_t matrixColwiseMax(const matrix_t& a)
    {
            return a.colwise().maxCoeff();
    }

    matrix_t matrixRowwiseMax(const matrix_t& a)
    {
            return a.rowwise().maxCoeff();
    }

    matrix_t matrixColwiseAdd(const matrix_t& a, const vector_t& b)
    {
            return a.colwise() + b;
    }

    void matrixColwiseAddInPlace(matrix_t& a, const vector_t& b)
    {
            a.colwise() += b;
    }

    matrix_t matrixRowwiseAdd(const matrix_t& a, const vector_t& b)
    {
            return a.rowwise() + b.transpose();
    }

    void matrixRowwiseAddInPlace(matrix_t& a, const vector_t& b)
    {
            a.rowwise() += b.transpose();
    }

    numeric_t matrixSum(const matrix_t& a)
    {
            return a.sum();
    }

    numeric_t matrixMean(const matrix_t& a)
    {
            return a.mean();
    }
    
    numeric_t matrixMin(const matrix_t& a)
    {
            return a.minCoeff();
    }

    numeric_t matrixMax(const matrix_t& a)
    {
            return a.maxCoeff();
    }

    size_t matrixRows(const matrix_t& a)
    {
            return a.rows();
    }

    size_t matrixCols(const matrix_t& a)
    {
            return a.cols();
    }

    matrix_t matrixRow(const matrix_t& a, size_t row)
    {
            return a.row(row);
    }

    matrix_t matrixCol(const matrix_t& a, size_t col)
    {
            return a.col(col);
    }

    numeric_t matrixAt(const matrix_t& a, size_t row, size_t col)
    {
            return a(row, col);
    }

    void matrixSetRow(matrix_t& a, size_t row, const vector_t& values)
    {
            a.row(row) = values;
    }

    void matrixSetCol(matrix_t& a, size_t col, const vector_t& values)
    {
            a.col(col) = values;
    }

    void matrixSetAt(matrix_t& a, size_t row, size_t col, numeric_t value)
    {
            a(row, col) = value;
    }

    /* vector operations */

    vector_t createVector(size_t size, numeric_t initialValue)
    {
            return vector_t::Constant(size, initialValue);
    }

    vector_t createVector(std::initializer_list<numeric_t> values)
    {
            return Eigen::Map<const vector_t>(values.begin(), values.size());
    }

    vector_t vectorAdd(const vector_t& a, const vector_t& b)
    {
            return a + b;
    }

    void vectorAddInPlace(vector_t& a, const vector_t& b)
    {
            a += b;
    }


    vector_t vectorScalarMultiply(const vector_t& a, numeric_t scalar)
    {
            return a * scalar;
    }

    void vectorScalarMultiplyInPlace(vector_t& a, numeric_t scalar)
    {
            a *= scalar;
    }

    vector_t vectorMultiply(const vector_t& a, const vector_t& b)
    {
            return a.cwiseProduct(b);
    }

    void vectorMultiplyInPlace(vector_t& a, const vector_t& b)
    {
            a = a.cwiseProduct(b);
    }

    numeric_t vectorDot(const vector_t& a, const vector_t& b)
    {
            return a.dot(b);
    }

    void vectorDotInPlace(numeric_t& result, const vector_t& a, const vector_t& b)
    {
            result = a.dot(b);
    }

    vector_t vectorSubtract(const vector_t& a, const vector_t& b)
    {
            return a - b;
    }

    void vectorSubtractInPlace(vector_t& a, const vector_t& b)
    {
            a -= b;
    }

    vector_t vectorDivide(const vector_t& a, const vector_t& b)
    {
            return a.cwiseQuotient(b);
    }

    void vectorDivideInPlace(vector_t& a, const vector_t& b)
    {
            a = a.cwiseQuotient(b);
    }

    vector_t vectorScalarDivide(const vector_t& a, numeric_t scalar)
    {
            return a / scalar;
    }

    void vectorScalarDivideInPlace(vector_t& a, numeric_t scalar)
    {
            a /= scalar;
    }

    numeric_t vectorSum(const vector_t& a)
    {
            return a.sum();
    }

    size_t vectorSize(const vector_t& a)
    {
            return a.size();
    }

    numeric_t vectorAt(const vector_t& a, size_t index)
    {
            return a[index];
    }
}
#endif
