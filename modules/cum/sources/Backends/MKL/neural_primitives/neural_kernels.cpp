/* Created by jaro on 7/19/26. */

#include <oneapi/mkl/blas.hpp>
#include <sycl/sycl.hpp>

#include "cum/functions/transform.hpp"

#include "internal/cumMKL.hpp"
#include "cum/neural_primitives/neural_kernels.hpp"

#include "cum/runtime.hpp"
#include "cum/functions/exponential.hpp"
#include "cum/functions/logistic.hpp"
#include "cum/functions/various.hpp"

#define USE_MKL_BLAS // I will add option for use oneDNN for fused kernels in near future




// Fused Kernels for deep learning
namespace cum::neural_primitives::neural_kernels
{
    /* Y = WX+B
     * --------------------
     * W cols - in features
     * W rows - out features
     * X rows - in features
     * X cols - batch size (currently not supported)
     * B rows - out features
     */


    void feed_forward(cumeric_t* Y, const cumeric_t* W, const cumeric_t* X, const cumeric_t* B, const std::size_t in_features, const std::size_t out_features, const dim_t batch, const functions::function_id activation)
    {
        // sycl::event copy_event = internal::getQueue().copy(B, Y, out_features);
        // copy_event.wait();

        int64_t m = out_features;
        int64_t n = 1; // fixed batch size
        int64_t k = in_features;
        sycl::event gemm_event = oneapi::mkl::blas::row_major::gemm(internal::getQueue(), // for some reason if we add gemm result to Y and
                                           oneapi::mkl::transpose::nontrans, // transpose W
                                           oneapi::mkl::transpose::nontrans, // transpose X
                                           m, n, k, 1.0, W,
                                           k, X, n, 1.0, Y,
                                           // n, {copy_event});
                                           n, oneapi::mkl::blas::compute_mode::standard,  {});
                                           // n, oneapi::mkl::blas::compute_mode::standard,  {copy_event});

        gemm_event.wait();

        internal::getQueue().parallel_for(sycl::range<1>(out_features), [=](sycl::id<1> idx) -> void
        {
            Y[idx] += B[idx];
        });

        //
        // if (activation == functions::function_id::linear)
            // return;

        functions::transform(Y, Y, out_features, activation);

    }

    void feed_forward(cumeric_t* Y, const cumeric_t* W, const cumeric_t* X, const cumeric_t* B, const std::size_t in_features, const std::size_t out_features, const functions::function_id activation)
    {
        feed_forward(Y, W, X, B, in_features, out_features, 1, activation);
    }

    void feed_forward_cached_raw(cumeric_t* Y, cumeric_t* R, const cumeric_t* W, const cumeric_t* X, const cumeric_t* B, const std::size_t in_features, const std::size_t out_features, const size_t batch, const functions::function_id activation)
    {
        sycl::event event_to_wait;

        if (batch == 1)
        {
            event_to_wait = internal::getQueue().copy(B, R, out_features);
        }
        else
        {
            event_to_wait = internal::getQueue().parallel_for(sycl::range<1>(out_features * batch), [=](sycl::id<1> idx) -> void
            {
                // R[idx] = B[idx % out_features];
                const std::size_t row = idx / batch;
                R[idx] = B[row];
            });
        }

        event_to_wait.wait();

        int64_t m = out_features;
        int64_t n = batch;
        int64_t k = in_features;
        sycl::event gemm_event = oneapi::mkl::blas::row_major::gemm(internal::getQueue(),
                                           oneapi::mkl::transpose::nontrans, // transpose W
                                           oneapi::mkl::transpose::nontrans, // transpose X
                                           m, n, k,
                                           1.0, W,k,
                                           X, n, 1.0,
                                           R, n,
                                           oneapi::mkl::blas::compute_mode::standard,  {event_to_wait});


        // tymczasowo fixed tanh dla debugowania przyklładu examples
        oneapi::mkl::vm::tanh(internal::getQueue(), m * n, R, Y, {gemm_event});
        // functions::transform(Y, R, out_features * batch, activation);
        runtime::sync();
    }

    void feed_forward_cached_raw(cumeric_t* Y, cumeric_t* R, const cumeric_t* W, const cumeric_t* X, const cumeric_t* B, const std::size_t in_features, const std::size_t out_features, const functions::function_id activation)
    {
        feed_forward_cached_raw(Y, R, W, X, B, in_features, out_features, 1, activation);
    }

    // void BCE(cumeric_t* grad, const cumeric_t* P, const cumeric_t* Y, const dim_t N)
    void BCE(cumeric_t* grad, cumeric_t* loss, const cumeric_t* P, const cumeric_t* Y, const dim_t N, const dim_t batch)
    {
        constexpr cumeric_t eps = static_cast<cumeric_t>(1e-7);

        auto reduction = sycl::reduction(loss, static_cast<cumeric_t>(0.0), sycl::plus<>());

        internal::getQueue().parallel_for(sycl::range<2>(N, batch), reduction, [=](sycl::id<2> idx, auto& sum) -> void
        {
            size_t i = idx[0];
            size_t sample = idx[1];
            size_t offset = i + sample * N;

            cumeric_t p = sycl::clamp(P[offset], eps, static_cast<cumeric_t>(1) - eps); // predicted value
            cumeric_t y = sycl::clamp(Y[offset], eps, static_cast<cumeric_t>(1) - eps); // predicted value
            // cumeric_t y = Y[offset]; // target value
            sum += -(y * sycl::log(p) + (1 - y) * sycl::log(1 - p));
            grad[offset] = ((1 - y) / (1 - p) - y / p);
        }).wait();
        loss[0] /= static_cast<cumeric_t>(batch);
    }
}