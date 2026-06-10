#include "cum/LinearAlgebra.hpp"
#include "cumMKL.hpp"
#include <cstring>
#include <oneapi/mkl/blas.hpp>

namespace cum::LinearAlgebra
{
    void add(cumeric_t* r, const cumeric_t* v, const cumeric_t* u, size_t N)
    {
        auto& q = library::getQueue();
        // auto copy = q.copy(v, r, N);
        // oneapi::mkl::blas::row_major::axpy(q, N, 1.0, u, 1, r, 1, {});
        oneapi::mkl::vm::add(q, 
                             N, // length of vectors 
                             v, // input vector
                             u, // input vector
                             r);// output vector
        q.wait();
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
        // oneapi::mkl::vm::add(q, N, r, v, a);
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
        q.wait();
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
        #if defined(CUM_USE_F64) || defined(CUM_USE_F32)
        {
            oneapi::mkl::blas::row_major::iamax(q, N, v, 1, &i);
        }
        #else // use custom argmax kernel
        {
            q.submit([=](sycl::handler& h){

                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
                    // This is a very naive implementation of argmax, it should be replaced with a more efficient one
                    if (v[i] > v[0]) {
                        i = i;
                    }
                });
            }).wait();
        }
        #endif
        r[0] = v[i];
        q.wait();
    }

    void argmin(cumeric_t* r, const cumeric_t* v, std::size_t N)
    {
        auto& q = library::getQueue();
        // oneapi::mkl::blas::row_major::iamin(q, N, v, 1, r, {});
        int64_t i;
        #if defined(CUM_USE_F64) || defined(CUM_USE_F32)
        {
            oneapi::mkl::blas::row_major::iamin(q, N, v, 1, &i);
        }
        #else // use custom argmin kernel
        {
            q.submit([=](sycl::handler& h){

                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
                    // This is a very naive implementation of argmin, it should be replaced with a more efficient one
                    if (v[i] < v[0]) {
                        i = i;
                    }
                });
            }).wait();
        }
        #endif
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
        oneapi::mkl::blas::row_major::gemm(q, 
                                           oneapi::mkl::transpose::nontrans, // transpose A
                                           oneapi::mkl::transpose::nontrans, // transpose B
                                           m, n, k, 1.0, A, // m=matrix A rows, n=matrix B cols, k=matrix A cols = matrix B rows
                                           k, B, n, 1.0, C, // k=matrix A cols = matrix B rows, n=matrix B cols, n=matrix C cols
                                           n, {});          //
        q.wait();
    }

    void matMulInPlace(cumeric_t* A, const cumeric_t* B, std::size_t m, std::size_t n, std::size_t k)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::gemm(q, 
                                           oneapi::mkl::transpose::nontrans, // transpose A
                                           oneapi::mkl::transpose::nontrans, // transpose B
                                           m, n, k, 1.0, A, // m=matrix A rows, n=matrix B cols, k=matrix A cols = matrix B rows
                                           k, B, n, 1.0, A, // k=matrix A cols = matrix B rows, n=matrix B cols, n=matrix A cols
                                           n, {});          //
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
        #if defined(CUM_USE_F64) || defined(CUM_USE_F32)
        {
            oneapi::mkl::blas::row_major::omatcopy(q, oneapi::mkl::transpose::trans, rows, cols, 1.0, B, cols, A, rows);
        }
        #else // use custom transposition kernel
        {
            q.submit([=](sycl::handler& h){
                h.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> i){
                    A[i[1] * rows + i[0]] = B[i[0] * cols + i[1]];
                });
            });
        }
        #endif
        q.wait();
    }

    void transposeInPlace(cumeric_t* mat, std::size_t rows, std::size_t cols)
    {
        auto& q = library::getQueue();
        #if defined(CUM_USE_F64) || defined(CUM_USE_F32)
        {
            oneapi::mkl::blas::row_major::imatcopy(q, oneapi::mkl::transpose::trans, rows, cols, 1.0, mat, cols, rows);
        }
        #else // use custom transposition kernel
        {
            q.submit([=](sycl::handler& h){
                h.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> i){
                    
                    const std::size_t r = i[0];
                    const std::size_t c = i[1];

                    if (r < c)
                    {
                        const std::size_t a = r * cols + c;
                        const std::size_t b = c * cols + r;

                        cumeric_t tmp = mat[a];
                        mat[a] = mat[b];
                        mat[b] = tmp;
                    }
                    
                });
            });
        }
        #endif
        q.wait();
    }

    void rowwiseSum(cumeric_t* r, const cumeric_t* mat, std::size_t rows, std::size_t cols)
    {
        auto& q = library::getQueue();
        #if defined(CUM_USE_F64) || defined(CUM_USE_F32)
        {
            oneapi::mkl::blas::row_major::gemv(q, oneapi::mkl::transpose::nontrans, rows, cols, 1.0, mat, cols, library::getOnes(), 1, 0.0, r, 1);
        }
        #else // use standard gemm
        {
            oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, 1, cols, rows, 1.0, mat, cols, library::getOnes(), 1, 0.0, r, cols);
        }
        #endif
        q.wait();
    }

    void colwiseSum(cumeric_t* r, const cumeric_t* mat, std::size_t rows, std::size_t cols)
    {
        auto& q = library::getQueue();
        #if defined(CUM_USE_F64) || defined(CUM_USE_F32)
        {
            oneapi::mkl::blas::row_major::gemv(q, oneapi::mkl::transpose::trans, rows, cols, 1.0, mat, cols, library::getOnes(), 1, 0.0, r, 1);
        }
        #else // use standard gemm
        {
            oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, cols, 1, rows, 1.0, mat, cols, library::getOnes(), 1, 0.0, r, 1);
        }
        #endif
        q.wait();
    }

    // void transform(cumeric_t* mat, const cumeric_t* old, void (*func)(cumeric_t* data, const std::size_t size), std::size_t N)
    // {
    //     auto& q = library::getQueue();
    //     cumeric_t* temp = (cumeric_t*)calloc(N, sizeof(cumeric_t));
    //     q.submit([=](sycl::handler& h){
    //         h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
    //             temp[i] = old[i];
    //         });
    //     }).wait();
    //     func(temp, N);
    //     q.submit([=](sycl::handler& h){
    //         func(temp, N);
    //     }).wait();
    //     free(temp);
    // }
    // void transformInPlace(cumeric_t* mat, void (*func)(cumeric_t* data, const std::size_t size), std::size_t N)
    // {
    //     auto& q = library::getQueue();
    //     q.submit([=](sycl::handler& h){
    //         func(mat, N);
    //     }).wait();
    // }

    // void transform(cumeric_t* mat, const cumeric_t* old, cumeric_t (*func)(cumeric_t x), std::size_t N)
    // {
    //     auto& q = library::getQueue();
    //     cumeric_t* temp = (cumeric_t*)calloc(N, sizeof(cumeric_t));
    //     q.submit([=](sycl::handler& h){
    //         h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
    //             temp[i] = func(old[i]);
    //         });
    //     }).wait();
    //     q.submit([=](sycl::handler& h){
    //         h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
    //             mat[i] = temp[i];
    //         });
    //     }).wait();
    //     free(temp);
    // }

    // void transformInPlace(cumeric_t* mat, cumeric_t (*func)(cumeric_t x), std::size_t N)
    // {
    //     auto& q = library::getQueue();
    //     q.submit([=](sycl::handler& h){
    //         h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i){
    //             mat[i] = func(mat[i]);
    //         });
    //     }).wait();
    // }

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

    void squaredNorm(cumeric_t& r, const cumeric_t* v, std::size_t N)
    {
        auto& q = library::getQueue();
        oneapi::mkl::blas::row_major::dot(q, N, v, 1, v, 1, &r);
        // oneapi::mkl::blas::row_major::n(q, N, v, 1, v, 1, &r);
        q.wait();
    }

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

    void normalizeInPlace(cumeric_t* v, std::size_t N) // cowise division by argmax
    {
        auto& q = library::getQueue();
        cumeric_t max = 0;
        LinearAlgebra::argmax(&max, v, N);
        cumeric_t* temp = (cumeric_t*)calloc(N, sizeof(cumeric_t));
        scale(temp, library::getOnes(), max, N);
        // q.submit([=](sycl::handler& h){
            oneapi::mkl::vm::div(q, N, v, temp, v, {});
        // });
        free(temp);
        q.wait();
    }
} // namespace cum
