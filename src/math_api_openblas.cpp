#ifdef _USE_OPENBLAS
#include "math_api.hpp"

#include <algorithm>
#include <random>
#include <cassert>

namespace YANN::math_api
{

static inline size_t matrixIndex(size_t row, size_t col, size_t cols)
{
    return row * cols + col;
}

static inline matrix_t allocateMatrix(size_t rows, size_t cols, numeric_t value = numeric_t())
{
    matrix_t mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = new numeric_t[rows * cols];
    for (size_t i = 0; i < rows * cols; ++i)
        mat.data[i] = value;
    return mat;
}

static inline vector_t allocateVector(size_t size, numeric_t value = numeric_t())
{
    vector_t vec;
    vec.size = size;
    vec.data = new numeric_t[size];
    for (size_t i = 0; i < size; ++i)
        vec.data[i] = value;
    return vec;
}

matrix_t createMatrix(size_t rows, size_t cols)
{
    return allocateMatrix(rows, cols, numeric_t());
}

matrix_t createMatrix(size_t rows, size_t cols, numeric_t initialValue)
{
    return allocateMatrix(rows, cols, initialValue);
}

matrix_t createMatrix(std::initializer_list<std::initializer_list<numeric_t>> values)
{
    size_t rows = values.size();
    size_t cols = rows > 0 ? values.begin()->size() : 0;
    matrix_t mat = allocateMatrix(rows, cols);
    size_t r = 0;
    for (const auto& row : values) {
        size_t c = 0;
        for (const auto& value : row) {
            mat.data[matrixIndex(r, c, cols)] = value;
            ++c;
        }
        ++r;
    }
    return mat;
}

matrix_t createRandomMatrix(size_t rows, size_t cols, numeric_t minValue, numeric_t maxValue)
{
    matrix_t mat = allocateMatrix(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(static_cast<double>(minValue), static_cast<double>(maxValue));
    for (size_t i = 0; i < rows * cols; ++i)
        mat.data[i] = static_cast<numeric_t>(dis(gen));
    return mat;
}

matrix_t matrixAdd(const matrix_t& a, const matrix_t& b)
{
    assert(a.rows == b.rows && a.cols == b.cols);
    matrix_t result = allocateMatrix(a.rows, a.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        result.data[i] = a.data[i] + b.data[i];
    return result;
}

void matrixAddInPlace(matrix_t& a, const matrix_t& b)
{
    assert(a.rows == b.rows && a.cols == b.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        a.data[i] += b.data[i];
}

matrix_t matrixMultiply(const matrix_t& a, const matrix_t& b)
{
    assert(a.cols == b.rows);
    matrix_t result = allocateMatrix(a.rows, b.cols);

#if defined(_USE_DOUBLE_PRECISION)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(a.rows), static_cast<int>(b.cols), static_cast<int>(a.cols),
                1.0,
                a.data, static_cast<int>(a.cols),
                b.data, static_cast<int>(b.cols),
                0.0,
                result.data, static_cast<int>(result.cols));
#elif defined(_USE_SINGLE_PRECISION)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(a.rows), static_cast<int>(b.cols), static_cast<int>(a.cols),
                1.0f,
                a.data, static_cast<int>(a.cols),
                b.data, static_cast<int>(b.cols),
                0.0f,
                result.data, static_cast<int>(result.cols));
#else
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < b.cols; ++j) {
            numeric_t sum = numeric_t();
            for (size_t k = 0; k < a.cols; ++k) {
                sum += a.data[matrixIndex(i, k, a.cols)] * b.data[matrixIndex(k, j, b.cols)];
            }
            result.data[matrixIndex(i, j, result.cols)] = sum;
        }
    }
#endif
    return result;
}

void matrixMultiplyInPlace(matrix_t& a, const matrix_t& b)
{
    matrix_t result = matrixMultiply(a, b);
    delete[] a.data;
    a = result;
}

matrix_t matrixScalarMultiply(const matrix_t& a, numeric_t scalar)
{
    matrix_t result = allocateMatrix(a.rows, a.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        result.data[i] = a.data[i] * scalar;
    return result;
}

void matrixScalarMultiplyInPlace(matrix_t& a, numeric_t scalar)
{
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        a.data[i] *= scalar;
}

matrix_t matrixElementwiseMultiply(const matrix_t& a, const matrix_t& b)
{
    assert(a.rows == b.rows && a.cols == b.cols);
    matrix_t result = allocateMatrix(a.rows, a.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        result.data[i] = a.data[i] * b.data[i];
    return result;
}

void matrixElementwiseMultiplyInPlace(matrix_t& a, const matrix_t& b)
{
    assert(a.rows == b.rows && a.cols == b.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        a.data[i] *= b.data[i];
}

matrix_t matrixSubtract(const matrix_t& a, const matrix_t& b)
{
    assert(a.rows == b.rows && a.cols == b.cols);
    matrix_t result = allocateMatrix(a.rows, a.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        result.data[i] = a.data[i] - b.data[i];
    return result;
}

void matrixSubtractInPlace(matrix_t& a, const matrix_t& b)
{
    assert(a.rows == b.rows && a.cols == b.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        a.data[i] -= b.data[i];
}

matrix_t matrixDivide(const matrix_t& a, const matrix_t& b)
{
    assert(a.rows == b.rows && a.cols == b.cols);
    matrix_t result = allocateMatrix(a.rows, a.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        result.data[i] = a.data[i] / b.data[i];
    return result;
}

void matrixDivideInPlace(matrix_t& a, const matrix_t& b)
{
    assert(a.rows == b.rows && a.cols == b.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        a.data[i] /= b.data[i];
}

void matrixElementwiseDivideInPlace(matrix_t& a, const matrix_t& b)
{
    assert(a.rows == b.rows && a.cols == b.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        a.data[i] /= b.data[i];
}

matrix_t matrixScalarDivide(const matrix_t& a, numeric_t scalar)
{
    matrix_t result = allocateMatrix(a.rows, a.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        result.data[i] = a.data[i] / scalar;
    return result;
}

void matrixScalarDivideInPlace(matrix_t& a, numeric_t scalar)
{
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        a.data[i] /= scalar;
}

matrix_t matrixTranspose(const matrix_t& a)
{
    matrix_t result = allocateMatrix(a.cols, a.rows);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            result.data[matrixIndex(j, i, result.cols)] = a.data[matrixIndex(i, j, a.cols)];
        }
    }
    return result;
}

void matrixTransposeInPlace(matrix_t& a)
{
    matrix_t transposed = matrixTranspose(a);
    delete[] a.data;
    a = transposed;
}

matrix_t matrixTransform(const matrix_t& a, numeric_t (*func)(numeric_t))
{
    matrix_t result = allocateMatrix(a.rows, a.cols);
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        result.data[i] = func(a.data[i]);
    return result;
}

void matrixTransformInPlace(matrix_t& a, numeric_t (*func)(numeric_t))
{
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        a.data[i] = func(a.data[i]);
}

matrix_t matrixNormalize(const matrix_t& a)
{
    assert(a.rows > 0 && a.cols > 0);
    size_t count = a.rows * a.cols;
    numeric_t minValue = a.data[0];
    numeric_t maxValue = a.data[0];
    for (size_t i = 1; i < count; ++i) {
        minValue = std::min(minValue, a.data[i]);
        maxValue = std::max(maxValue, a.data[i]);
    }
    numeric_t range = maxValue - minValue;
    matrix_t result = allocateMatrix(a.rows, a.cols);
    for (size_t i = 0; i < count; ++i)
        result.data[i] = range == numeric_t() ? numeric_t() : (a.data[i] - minValue) / range;
    return result;
}

void matrixNormalizeInPlace(matrix_t& a)
{
    assert(a.rows > 0 && a.cols > 0);
    size_t count = a.rows * a.cols;
    numeric_t minValue = a.data[0];
    numeric_t maxValue = a.data[0];
    for (size_t i = 1; i < count; ++i) {
        minValue = std::min(minValue, a.data[i]);
        maxValue = std::max(maxValue, a.data[i]);
    }
    numeric_t range = maxValue - minValue;
    for (size_t i = 0; i < count; ++i)
        a.data[i] = range == numeric_t() ? numeric_t() : (a.data[i] - minValue) / range;
}

matrix_t matrixColwise(const matrix_t& a, numeric_t (*func)(const vector_t&))
{
    matrix_t result = allocateMatrix(1, a.cols);
    for (size_t col = 0; col < a.cols; ++col) {
        vector_t column = allocateVector(a.rows);
        for (size_t row = 0; row < a.rows; ++row)
            column.data[row] = a.data[matrixIndex(row, col, a.cols)];
        result.data[col] = func(column);
        delete[] column.data;
    }
    return result;
}

matrix_t matrixRowwise(const matrix_t& a, numeric_t (*func)(const vector_t&))
{
    matrix_t result = allocateMatrix(a.rows, 1);
    for (size_t row = 0; row < a.rows; ++row) {
        vector_t rowVec = allocateVector(a.cols);
        for (size_t col = 0; col < a.cols; ++col)
            rowVec.data[col] = a.data[matrixIndex(row, col, a.cols)];
        result.data[row] = func(rowVec);
        delete[] rowVec.data;
    }
    return result;
}

matrix_t matrixColwiseSum(const matrix_t& a)
{
    matrix_t result = allocateMatrix(1, a.cols);
    for (size_t col = 0; col < a.cols; ++col) {
        numeric_t sum = numeric_t();
        for (size_t row = 0; row < a.rows; ++row)
            sum += a.data[matrixIndex(row, col, a.cols)];
        result.data[col] = sum;
    }
    return result;
}

matrix_t matrixRowwiseSum(const matrix_t& a)
{
    matrix_t result = allocateMatrix(a.rows, 1);
    for (size_t row = 0; row < a.rows; ++row) {
        numeric_t sum = numeric_t();
        for (size_t col = 0; col < a.cols; ++col)
            sum += a.data[matrixIndex(row, col, a.cols)];
        result.data[row] = sum;
    }
    return result;
}

matrix_t matrixColwiseMean(const matrix_t& a)
{
    matrix_t result = matrixColwiseSum(a);
    numeric_t divisor = static_cast<numeric_t>(a.rows);
    for (size_t i = 0; i < result.cols; ++i)
        result.data[i] /= divisor;
    return result;
}

matrix_t matrixRowwiseMean(const matrix_t& a)
{
    matrix_t result = matrixRowwiseSum(a);
    numeric_t divisor = static_cast<numeric_t>(a.cols);
    for (size_t i = 0; i < result.rows; ++i)
        result.data[i] /= divisor;
    return result;
}

matrix_t matrixColwiseMin(const matrix_t& a)
{
    matrix_t result = allocateMatrix(1, a.cols);
    for (size_t col = 0; col < a.cols; ++col) {
        numeric_t minValue = a.data[matrixIndex(0, col, a.cols)];
        for (size_t row = 1; row < a.rows; ++row)
            minValue = std::min(minValue, a.data[matrixIndex(row, col, a.cols)]);
        result.data[col] = minValue;
    }
    return result;
}

matrix_t matrixRowwiseMin(const matrix_t& a)
{
    matrix_t result = allocateMatrix(a.rows, 1);
    for (size_t row = 0; row < a.rows; ++row) {
        numeric_t minValue = a.data[matrixIndex(row, 0, a.cols)];
        for (size_t col = 1; col < a.cols; ++col)
            minValue = std::min(minValue, a.data[matrixIndex(row, col, a.cols)]);
        result.data[row] = minValue;
    }
    return result;
}

matrix_t matrixColwiseMax(const matrix_t& a)
{
    matrix_t result = allocateMatrix(1, a.cols);
    for (size_t col = 0; col < a.cols; ++col) {
        numeric_t maxValue = a.data[matrixIndex(0, col, a.cols)];
        for (size_t row = 1; row < a.rows; ++row)
            maxValue = std::max(maxValue, a.data[matrixIndex(row, col, a.cols)]);
        result.data[col] = maxValue;
    }
    return result;
}

matrix_t matrixRowwiseMax(const matrix_t& a)
{
    matrix_t result = allocateMatrix(a.rows, 1);
    for (size_t row = 0; row < a.rows; ++row) {
        numeric_t maxValue = a.data[matrixIndex(row, 0, a.cols)];
        for (size_t col = 1; col < a.cols; ++col)
            maxValue = std::max(maxValue, a.data[matrixIndex(row, col, a.cols)]);
        result.data[row] = maxValue;
    }
    return result;
}

matrix_t matrixColwiseAdd(const matrix_t& a, const vector_t& b)
{
    assert(b.size == a.cols);
    matrix_t result = allocateMatrix(a.rows, a.cols);
    for (size_t row = 0; row < a.rows; ++row) {
        for (size_t col = 0; col < a.cols; ++col)
            result.data[matrixIndex(row, col, a.cols)] = a.data[matrixIndex(row, col, a.cols)] + b.data[col];
    }
    return result;
}

void matrixColwiseAddInPlace(matrix_t& a, const vector_t& b)
{
    assert(b.size == a.cols);
    for (size_t row = 0; row < a.rows; ++row) {
        for (size_t col = 0; col < a.cols; ++col)
            a.data[matrixIndex(row, col, a.cols)] += b.data[col];
    }
}

matrix_t matrixRowwiseAdd(const matrix_t& a, const vector_t& b)
{
    assert(b.size == a.rows);
    matrix_t result = allocateMatrix(a.rows, a.cols);
    for (size_t row = 0; row < a.rows; ++row) {
        for (size_t col = 0; col < a.cols; ++col)
            result.data[matrixIndex(row, col, a.cols)] = a.data[matrixIndex(row, col, a.cols)] + b.data[row];
    }
    return result;
}

void matrixRowwiseAddInPlace(matrix_t& a, const vector_t& b)
{
    assert(b.size == a.rows);
    for (size_t row = 0; row < a.rows; ++row) {
        for (size_t col = 0; col < a.cols; ++col)
            a.data[matrixIndex(row, col, a.cols)] += b.data[row];
    }
}

numeric_t matrixSum(const matrix_t& a)
{
    numeric_t sum = numeric_t();
    size_t count = a.rows * a.cols;
    for (size_t i = 0; i < count; ++i)
        sum += a.data[i];
    return sum;
}

numeric_t matrixMean(const matrix_t& a)
{
    return matrixSum(a) / static_cast<numeric_t>(a.rows * a.cols);
}

numeric_t matrixMin(const matrix_t& a)
{
    assert(a.rows > 0 && a.cols > 0);
    numeric_t minValue = a.data[0];
    size_t count = a.rows * a.cols;
    for (size_t i = 1; i < count; ++i)
        minValue = std::min(minValue, a.data[i]);
    return minValue;
}

numeric_t matrixMax(const matrix_t& a)
{
    assert(a.rows > 0 && a.cols > 0);
    numeric_t maxValue = a.data[0];
    size_t count = a.rows * a.cols;
    for (size_t i = 1; i < count; ++i)
        maxValue = std::max(maxValue, a.data[i]);
    return maxValue;
}

size_t matrixRows(const matrix_t& a)
{
    return a.rows;
}

size_t matrixCols(const matrix_t& a)
{
    return a.cols;
}

matrix_t matrixRow(const matrix_t& a, size_t row)
{
    assert(row < a.rows);
    matrix_t result = allocateMatrix(1, a.cols);
    for (size_t col = 0; col < a.cols; ++col)
        result.data[col] = a.data[matrixIndex(row, col, a.cols)];
    return result;
}

matrix_t matrixCol(const matrix_t& a, size_t col)
{
    assert(col < a.cols);
    matrix_t result = allocateMatrix(a.rows, 1);
    for (size_t row = 0; row < a.rows; ++row)
        result.data[row] = a.data[matrixIndex(row, col, a.cols)];
    return result;
}

numeric_t matrixAt(const matrix_t& a, size_t row, size_t col)
{
    assert(row < a.rows && col < a.cols);
    return a.data[matrixIndex(row, col, a.cols)];
}

void matrixSetRow(matrix_t& a, size_t row, const vector_t& values)
{
    assert(row < a.rows && values.size == a.cols);
    for (size_t col = 0; col < a.cols; ++col)
        a.data[matrixIndex(row, col, a.cols)] = values.data[col];
}

void matrixSetCol(matrix_t& a, size_t col, const vector_t& values)
{
    assert(col < a.cols && values.size == a.rows);
    for (size_t row = 0; row < a.rows; ++row)
        a.data[matrixIndex(row, col, a.cols)] = values.data[row];
}

void matrixSetAt(matrix_t& a, size_t row, size_t col, numeric_t value)
{
    assert(row < a.rows && col < a.cols);
    a.data[matrixIndex(row, col, a.cols)] = value;
}

vector_t createVector(size_t size)
{
    return allocateVector(size, numeric_t());
}

vector_t createVector(size_t size, numeric_t initialValue)
{
    return allocateVector(size, initialValue);
}

vector_t createVector(std::initializer_list<numeric_t> values)
{
    vector_t vec = allocateVector(values.size());
    size_t i = 0;
    for (const auto& value : values)
        vec.data[i++] = value;
    return vec;
}

vector_t vectorAdd(const vector_t& a, const vector_t& b)
{
    assert(a.size == b.size);
    vector_t result = allocateVector(a.size);
    for (size_t i = 0; i < a.size; ++i)
        result.data[i] = a.data[i] + b.data[i];
    return result;
}

void vectorAddInPlace(vector_t& a, const vector_t& b)
{
    assert(a.size == b.size);
    for (size_t i = 0; i < a.size; ++i)
        a.data[i] += b.data[i];
}

vector_t vectorMultiply(const vector_t& a, const vector_t& b)
{
    assert(a.size == b.size);
    vector_t result = allocateVector(a.size);
    for (size_t i = 0; i < a.size; ++i)
        result.data[i] = a.data[i] * b.data[i];
    return result;
}

void vectorMultiplyInPlace(vector_t& a, const vector_t& b)
{
    assert(a.size == b.size);
    for (size_t i = 0; i < a.size; ++i)
        a.data[i] *= b.data[i];
}

vector_t vectorScalarMultiply(const vector_t& a, numeric_t scalar)
{
    vector_t result = allocateVector(a.size);
    for (size_t i = 0; i < a.size; ++i)
        result.data[i] = a.data[i] * scalar;
    return result;
}

void vectorScalarMultiplyInPlace(vector_t& a, numeric_t scalar)
{
    for (size_t i = 0; i < a.size; ++i)
        a.data[i] *= scalar;
}

numeric_t vectorDot(const vector_t& a, const vector_t& b)
{
    assert(a.size == b.size);
    numeric_t result = numeric_t();
    for (size_t i = 0; i < a.size; ++i)
        result += a.data[i] * b.data[i];
    return result;
}

void vectorDotInPlace(numeric_t& result, const vector_t& a, const vector_t& b)
{
    result = vectorDot(a, b);
}

vector_t vectorSubtract(const vector_t& a, const vector_t& b)
{
    assert(a.size == b.size);
    vector_t result = allocateVector(a.size);
    for (size_t i = 0; i < a.size; ++i)
        result.data[i] = a.data[i] - b.data[i];
    return result;
}

void vectorSubtractInPlace(vector_t& a, const vector_t& b)
{
    assert(a.size == b.size);
    for (size_t i = 0; i < a.size; ++i)
        a.data[i] -= b.data[i];
}

vector_t vectorDivide(const vector_t& a, const vector_t& b)
{
    assert(a.size == b.size);
    vector_t result = allocateVector(a.size);
    for (size_t i = 0; i < a.size; ++i)
        result.data[i] = a.data[i] / b.data[i];
    return result;
}

void vectorDivideInPlace(vector_t& a, const vector_t& b)
{
    assert(a.size == b.size);
    for (size_t i = 0; i < a.size; ++i)
        a.data[i] /= b.data[i];
}

vector_t vectorElementwiseDivide(const vector_t& a, const vector_t& b)
{
    return vectorDivide(a, b);
}

void vectorElementwiseDivideInPlace(vector_t& a, const vector_t& b)
{
    vectorDivideInPlace(a, b);
}

vector_t vectorScalarDivide(const vector_t& a, numeric_t scalar)
{
    vector_t result = allocateVector(a.size);
    for (size_t i = 0; i < a.size; ++i)
        result.data[i] = a.data[i] / scalar;
    return result;
}

void vectorScalarDivideInPlace(vector_t& a, numeric_t scalar)
{
    for (size_t i = 0; i < a.size; ++i)
        a.data[i] /= scalar;
}

vector_t vectorTransform(const vector_t& a, numeric_t (*func)(numeric_t))
{
    vector_t result = allocateVector(a.size);
    for (size_t i = 0; i < a.size; ++i)
        result.data[i] = func(a.data[i]);
    return result;
}

void vectorTransformInPlace(vector_t& a, numeric_t (*func)(numeric_t))
{
    for (size_t i = 0; i < a.size; ++i)
        a.data[i] = func(a.data[i]);
}

numeric_t vectorSum(const vector_t& a)
{
    numeric_t sum = numeric_t();
    for (size_t i = 0; i < a.size; ++i)
        sum += a.data[i];
    return sum;
}

size_t vectorSize(const vector_t& a)
{
    return a.size;
}

numeric_t vectorAt(const vector_t& a, size_t index)
{
    assert(index < a.size);
    return a.data[index];
}

}
#endif
