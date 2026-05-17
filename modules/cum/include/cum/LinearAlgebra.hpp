#pragma once

#include "cum/Core.hpp"

namespace cum::LinearAlgebra /* LATK */
{
    void add(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, std::size_t N);
    void addInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N);
    
    void add(cumeric_t* r, const cumeric_t* v, const cumeric_t& a, std::size_t N);
    void addInPlace(cumeric_t* v, const cumeric_t& a, std::size_t N);

    void sum(cumeric_t& r, const cumeric_t* v, std::size_t N);

    void cwiseProduct(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, std::size_t N);
    void cwiseProductInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N);

    void scale(cumeric_t* r, const cumeric_t* v, const cumeric_t& a, const std::size_t n);
    void scaleInPlace(cumeric_t* v, const cumeric_t& a, const std::size_t n);

    void sub(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, std::size_t N);
    void subInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N);

    void div(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, std::size_t N);
    void divInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N);

    void argmax(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void argmin(cumeric_t* r, const cumeric_t* v, std::size_t N);

    void matAdd(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);
    void matAddInPlace(cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);

    void matSub(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);
    void matSubInPlace(cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);

    void matMul(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t m, std::size_t n, std::size_t k);
    void matMulInPlace(cumeric_t* A, const cumeric_t* B, std::size_t m, std::size_t n, std::size_t k);

    void matVecMul(cumeric_t* r, const cumeric_t* A, const cumeric_t* v, std::size_t rows, std::size_t cols);

    void matScale(cumeric_t* mat, const cumeric_t* old, const cumeric_t& a, const std::size_t& rows, const std::size_t& cols);
    void matScaleInPlace(cumeric_t* mat, const cumeric_t& a, const std::size_t& rows, const std::size_t& cols);

    void transpose(cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);
    void transposeInPlace(cumeric_t* mat, std::size_t rows, std::size_t cols);

    void rowwiseOpInPlace(cumeric_t* mat, const cumeric_t* arr, std::size_t rows, std::size_t cols, void (*op)(cumeric_t* row, const cumeric_t* v, const std::size_t cols));
    void colwiseOpInPlace(cumeric_t* mat, const cumeric_t* arr, std::size_t rows, std::size_t cols, void (*op)(cumeric_t* col, const cumeric_t* v, const std::size_t rows));

    void rowwiseSum(cumeric_t* r, const cumeric_t* mat, std::size_t rows, std::size_t cols);
    void colwiseSum(cumeric_t* r, const cumeric_t* mat, std::size_t rows, std::size_t cols);

    void transform(cumeric_t* mat, const cumeric_t* old, void (*func)(cumeric_t* data, const std::size_t size));
    void transformInPlace(cumeric_t* mat, void (*func)(cumeric_t* data, const std::size_t size));

    void transform(cumeric_t* mat, const cumeric_t* old, cumeric_t (*func)(cumeric_t x), std::size_t N);
    void transformInPlace(cumeric_t* mat, cumeric_t (*func)(cumeric_t x), std::size_t N);

    void sin(cumeric_t* r, const cumeric_t* v, const std::size_t& N);
    void sinInPlace(cumeric_t* v, const std::size_t& N);
    
    void cos(cumeric_t* r, const cumeric_t* v, const std::size_t& N);
    void cosInPlace(cumeric_t* v, const std::size_t& N);

    void tanh(cumeric_t* r, const cumeric_t* v, const std::size_t& N);
    void tanhInPlace(cumeric_t* v, const std::size_t& N);

    void exp(cumeric_t* r, const cumeric_t* v, const std::size_t& N);
    void expInPlace(cumeric_t* v, const std::size_t& N);

    void relu(cumeric_t* r, const cumeric_t* v, const std::size_t& N);
    void reluInPlace(cumeric_t* v, const std::size_t& N);

    void norm(cumeric_t& r, const cumeric_t* v, std::size_t N);
    void normInplace(cumeric_t* v, std::size_t N);

    void squaredNorm(cumeric_t& r, const cumeric_t* v, std::size_t N);

    void normalize(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void normalizeInPlace(cumeric_t* v, std::size_t N);
} // namespace cum
