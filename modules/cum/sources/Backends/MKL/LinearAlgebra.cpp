#include "cum/LinearAlgbebra.hpp"

#include <cstring>
#include "cumMKL.hpp"

#include <oneapi/mkl/blas.hpp>

namespace cum::LinearAlgebra
{
    void add(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, size_t N)
    {
        auto& q = library::getQueue();
        auto copy = q.copy(v, r, N);
        oneapi::mkl::blas::row_major::axpy(q, N, 1.0, u, 1, r, 1, {copy});
    }

    void sub(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, size_t N)
    {
        auto& q = library::getQueue();
        auto copy = q.copy(v, r, N);
        oneapi::mkl::blas::row_major::axpy(q, N, -1.0, u, 1, r, 1, {copy});
    }

    void addInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::axpy(q, N, 1.0, u, 1, v, 1);
        q.wait();
    }

    void subInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::axpy(q, N, -1.0, u, 1, v, 1);
        q.wait();
    }

    void matAdd(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols)
    {
        add(C, A, B, rows * cols);
    }

    void matAddInPlace(cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols)
    {
        addInPlace(A, B, rows * cols);
    }

    void matSub(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols)
    {
        add(C, A, B, rows * cols);
    }

    void matSubInPlace(cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols)
    {
        subInPlace(A, B, rows * cols);
    }

    void matMul(cumeric_t* C, const cumeric_t* A, const cumeric_t* B, std::size_t m, std::size_t n, std::size_t k)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, 1.0, A, k, B, n, 1.0, C, n, {});
        q.wait();
    }

} // namespace cum
