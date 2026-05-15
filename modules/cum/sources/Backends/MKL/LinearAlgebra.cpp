#include "cum/LinearAlgebra.hpp"
#include "cumMKL.hpp"
#include <cstring>
#include <oneapi/mkl/blas.hpp>

namespace cum::LinearAlgebra
{
    void add(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, size_t N)
    {
        auto& q = library::getQueue();
        auto copy = q.copy(v, r, N);
        oneapi::mkl::blas::row_major::axpy(q, N, 1.0, u, 1, r, 1, {copy});
    }

    void addInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::axpy(q, N, 1.0, u, 1, v, 1);
        q.wait();
    }

    void add(cumeric_t* r, const cumeric_t* v, const cumeric_t& a, std::size_t N)
    {
        auto& q = library::getQueue();
        q.submit([=](sycl::handler& h){

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
                r[i] = v[i] + a;
            });
        });
        q.wait();
    }

    void addInPlace(cumeric_t* v, const cumeric_t& a, std::size_t N)
    {
        auto& q = library::getQueue();
        q.submit([=](sycl::handler& h){

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
                v[i] += a;
            });
        });
        q.wait();
    }

    void sum(cumeric_t& r, const cumeric_t* v, std::size_t N)
    {
        auto& q = library::getQueue();
        // oneapi::mkl::blas::row_major::asum(q, N, v, 1, &r);
        oneapi::mkl::blas::row_major::dot(q, N, v, 1, library::getOnes(), 1, &r);
        q.wait();
    }    

    void cwiseProduct(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, std::size_t N)
    {
        auto& q = library::getQueue();
        q.submit([=](sycl::handler& h){

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
                r[i] = v[i] * u[i];
            });
        });
        q.wait();
    }

    void cwiseProductInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N)
    {
        auto& q = library::getQueue();
        q.submit([=](sycl::handler& h){

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
                v[i] *= u[i];
            });
        });
        q.wait();
    }

    void sub(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, size_t N)
    {
        auto& q = library::getQueue();
        auto copy = q.copy(v, r, N);
        oneapi::mkl::blas::row_major::axpy(q, N, -1.0, u, 1, r, 1, {copy});
    }

    void subInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::axpy(q, N, -1.0, u, 1, v, 1);
        q.wait();
    }

    void div(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, std::size_t N)
    {
        auto& q = library::getQueue();
        q.submit([=](sycl::handler& h){

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
                r[i] = v[i] / u[i];
            });
        });
        q.wait();
    }

    void divInPlace(cumeric_t* v, const cumeric_t* u, std::size_t N)
    {
        auto& q = library::getQueue();
        q.submit([=](sycl::handler& h){

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
                v[i] /= u[i];
            });
        });
        q.wait();
    }

    void argmax(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        auto& q = library::getQueue();
        int64_t i;
        oneapi::mkl::blas::row_major::iamax(q, N, v, 1, &i);
        r[0] = v[i];
        q.wait();
    }

    void argmin(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        auto& q = library::getQueue();
        // oneapi::mkl::blas::row_major::iamin(q, N, v, 1, r, {});
        int64_t i;
        oneapi::mkl::blas::row_major::iamin(q, N, v, 1, &i);
        r[0] = v[i];
        q.wait();
    }

    void scale(cumeric_t* r, const cumeric_t* v, const cumeric_t& a, const std::size_t n)
    {
        auto& q = library::getQueue();
        q.submit([=](sycl::handler& h){

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
                r[i] = v[i] * a;
            });
        });
        q.wait();
    }

    void scaleInPlace(cumeric_t* v, const cumeric_t& a, const std::size_t n)
    {
        auto& q = library::getQueue();
        q.submit([=](sycl::handler& h){

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){
                v[i] *= a;
            });
        });
        q.wait();
    }


    void norm(cumeric_t& r, const cumeric_t* v, std::size_t N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::nrm2(q, N, v, 1, &r);
        q.wait();
    }

    void normInplace(cumeric_t* v, std::size_t N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::nrm2(q, N, v, 1, v);
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
        sub(C, A, B, rows * cols);
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

    void matMulInPlace(cumeric_t* A, const cumeric_t* B, std::size_t m, std::size_t n, std::size_t k)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, m, n, k, 1.0, A, k, B, n, 1.0, A, n, {});
        q.wait();
    }

    void matScale(cumeric_t* mat, const cumeric_t* old, const cumeric_t& a, const std::size_t& rows, const std::size_t& cols)
    {
        scale(mat, old, a, rows * cols);
    }

    void matScaleInPlace(cumeric_t* mat, const cumeric_t& a, const std::size_t& rows, const std::size_t& cols)
    {
        scaleInPlace(mat, a, rows * cols);
    }

    void transpose(cumeric_t* A, const cumeric_t* B, std::size_t rows, std::size_t cols)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::omatcopy(q, oneapi::mkl::transpose::trans, rows, cols, 1.0, B, cols, A, rows);
        q.wait();
    }

    void transposeInPlace(cumeric_t* mat, std::size_t rows, std::size_t cols)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::imatcopy(q, oneapi::mkl::transpose::trans, rows, cols, 1.0, mat, cols, rows);
        q.wait();
    }

    void rowwiseSum(cumeric_t* r, const cumeric_t* mat, std::size_t rows, std::size_t cols)
    {
        auto& q = library::getQueue();
        // cumeric_t* temp = (cumeric_t*)calloc(cols, sizeof(cumeric_t));
        oneapi::mkl::blas::row_major::gemv(q, oneapi::mkl::transpose::nontrans, rows, cols, 1.0, mat, cols, library::getOnes(), 1, 0.0, r, 1);
        // free(temp);
        q.wait();
    }

    void colwiseSum(cumeric_t* r, const cumeric_t* mat, std::size_t rows, std::size_t cols)
    {
        auto& q = library::getQueue();
        // cumeric_t* temp = (cumeric_t*)calloc(rows, sizeof(cumeric_t));
        oneapi::mkl::blas::row_major::gemv(q, oneapi::mkl::transpose::trans, rows, cols, 1.0, mat, cols, library::getOnes(), 1, 0.0, r, 1);
        // free(temp);
        q.wait();
    }

    void sin(cumeric_t* r, const cumeric_t* v, const std::size_t& N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::vm::sin(q, N, v, r, {});
        q.wait();
    }

    void sinInPlace(cumeric_t* v, const std::size_t& N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::vm::sin(q, N, v, v, {});
        q.wait();
    }
    
    void cos(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::vm::cos(q, N, v, r, {});
        q.wait();
    }

    void cosInPlace(cumeric_t* v, const std::size_t& N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::vm::cos(q, N, v, v, {});
        q.wait();
    }

    void tanh(cumeric_t* r, const cumeric_t* v, const std::size_t& N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::vm::tanh(q, N, v, r, {});
        q.wait();
    }

    void tanhInPlace(cumeric_t* v, std::size_t& N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::vm::tanh(q, N, v, v, {});
        q.wait();
    }

    void exp(cumeric_t* r, const cumeric_t* v, const std::size_t& N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::vm::exp(q, N, v, r, {});
        q.wait();
    }

    void expInPlace(cumeric_t* v, const std::size_t& N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::vm::exp(q, N, v, v, {});
        q.wait();
    }

    void relu(cumeric_t* r, const cumeric_t* v, const std::size_t& N)
    {
        auto& q = library::getQueue();
        cumeric_t* temp = (cumeric_t*)calloc(N, sizeof(cumeric_t));
        oneapi::mkl::vm::fmax(q, N, v, temp, r, {});
        free(temp);
        q.wait();
    }

    void reluInPlace(cumeric_t* v, const std::size_t& N)
    {
        auto& q = library::getQueue();
        cumeric_t* temp = (cumeric_t*)calloc(N, sizeof(cumeric_t));
        oneapi::mkl::vm::fmax(q, N, v, temp, v, {});
        free(temp);
        q.wait();
    }

    // void norm(cumeric_t& r, const cumeric_t* v, std::size_t N)
    // {
    //     auto& q = library::getQueue();
    //     oneapi::mkl::blas::row_major::nrm2(q, N, v, 1, &r);
    //     q.wait();
    // }

    // void normInplace(cumeric_t* v, std::size_t N)
    // {
    //     auto& q = library::getQueue();
    //     oneapi::mkl::blas::row_major::nrm2(q, N, v, 1, v);
    //     q.wait();
    // }

    void normalize(cumeric_t* r, const cumeric_t* v, std::size_t N) // cowise division by argmax
    {
        auto& q = library::getQueue();
        cumeric_t max = 0;
        LinearAlgebra::argmax(&max, v, N);
        cumeric_t* temp = (cumeric_t*)calloc(N, sizeof(cumeric_t));
        scale(temp, library::getOnes(), max, N);
        // q.submit([=](sycl::handler& h){
        //     // oneapi::mkl::blas::row_major::scal(q, N, 1.0 / max, r, 1);
            oneapi::mkl::vm::div(q, N, v, temp, r, {});
        // });
        free(temp);
        q.wait();
    }
} // namespace cum
