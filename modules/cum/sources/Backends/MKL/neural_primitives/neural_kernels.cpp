/* Created by jaro on 7/19/26. */

#include <oneapi/mkl/blas.hpp>
#include <sycl/sycl.hpp>

#include "cum/functions/transform.hpp"

#include "internal/cumMKL.hpp"
#include "cum/neural_primitives/neural_kernels.hpp"

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
    void feed_forward(cumeric_t* Y, const cumeric_t* W, const cumeric_t* X, const cumeric_t* B, const std::size_t in_features, const std::size_t out_features, functions::function_id activation)
    {
        sycl::event copy_event = internal::getQueue().copy(B, Y, out_features);

        int64_t m = out_features;
        int64_t n = 1; // fixed batch size
        int64_t k = in_features;
        sycl::event gemm_event = oneapi::mkl::blas::row_major::gemm(internal::getQueue(),
                                           oneapi::mkl::transpose::nontrans, // transpose W
                                           oneapi::mkl::transpose::nontrans, // transpose X
                                           m, n, k, 1.0, W,
                                           k, X, n, 1.0, Y,
                                           n, {copy_event});
        gemm_event.wait();
        functions::transform(Y, Y, out_features, activation);
    }
}