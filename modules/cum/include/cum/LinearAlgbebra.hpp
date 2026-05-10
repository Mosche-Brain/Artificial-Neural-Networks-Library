#pragma once

#include "cum/Core.hpp"

namespace cum::LinearAlgebra
{
    void add(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, std::size_t N);
    void addInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N);

    void sub(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, std::size_t N);
    void subInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N);

    void matAdd(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);
    void matAddInPlace(cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);
    void matSub(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);
    void matSubInPlace(cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols);

    void matMul(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t m, std::size_t n, std::size_t k);

} // namespace cum
