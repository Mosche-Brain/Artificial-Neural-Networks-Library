#pragma once

#include "cum/Core.hpp"

namespace cum::LinearAlgebra
{
    void add(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, std::size_t N);
    void addInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N);
    
    void add(cumeric_t* r, const cumeric_t* v, const cumeric_t& a, std::size_t N);
    void addInPlace(cumeric_t* v, const cumeric_t& a, std::size_t N);

    void scale(cumeric_t* r, const cumeric_t* v, const cumeric_t& a, const std::size_t n);
    void scaleInPlace(cumeric_t* v, const cumeric_t& a, const std::size_t n);

    void sub(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, std::size_t N);
    void subInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N);

    void matAdd(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);
    void matAddInPlace(cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);
    void matSub(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);
    void matSubInPlace(cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);

    void matMul(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t m, std::size_t n, std::size_t k);
    void matMulInPlace(cumeric_t* A, const cumeric_t* B, std::size_t m, std::size_t n, std::size_t k);

    void matScale(cumeric_t* mat, const cumeric_t* old, const cumeric_t& a, const std::size_t& rows, const std::size_t& cols);
    void matScaleInPlace(cumeric_t* mat, const cumeric_t& a, const std::size_t& rows, const std::size_t& cols);

    void transpose(cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);
    void transposeInPlace(cumeric_t* mat, std::size_t rows, std::size_t cols);

    // void transform(cumeric_t* mat, const cumeric_t* old, )

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
} // namespace cum
