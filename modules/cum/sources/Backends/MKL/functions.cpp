#include "cum/functions.hpp"
#include "cum/LinearAlgebra.hpp"
#include "cum/memory.hpp"

#include "cumMKL.hpp"

#include <oneapi/mkl/vm/buffer.hpp>
#include <sycl/sycl.hpp>

#include <string.h>
#if defined(BUILD_DEBUG_OUTPUT)
    #include <iostream>
#endif

namespace cum::functions
{
    /* Standard Functions */
    cumeric_t linear(cumeric_t x) { return x; }
    cumeric_t linear_derivative(cumeric_t x) { return 1; }

    cumeric_t gelu(cumeric_t x)
    {
        #if defined(BUILD_USE_F64)
        #define M_SQRT_2_OVER_PI 0.7978845608028654
        #define M_SQRT_2_OVER_PI_A 0.044715000000000000
        #elif defined(BUILD_USE_F32)
        #define M_SQRT_2_OVER_PI 0.79788456f
        #define M_SQRT_2_OVER_PI_A 0.044715f
        #elif defined(BUILD_USE_F16)
        #define M_SQRT_2_OVER_PI 0.79788456_c
        #define M_SQRT_2_OVER_PI_A 0.044715_c
        #elif defined(BUILD_USE_BF16)
        #define M_SQRT_2_OVER_PI 0.79788456bf16
        #define M_SQRT_2_OVER_PI_A 0.044715bf16
        #else
        #define M_SQRT_2_OVER_PI 0.79788456f
        #define M_SQRT_2_OVER_PI_A 0.044715f
        #endif

        return 0.5_c * x * (1.0_c + tanhf(M_SQRT_2_OVER_PI * (x + M_SQRT_2_OVER_PI_A * x * x * x)));
    }
    cumeric_t GELU_derivative(cumeric_t x)
    {
        #if defined(BUILD_USE_F64)
        #define M_SQRT_2_OVER_PI 0.7978845608028654
        #define M_SQRT_2_OVER_PI_A 0.044715000000000000
        #elif defined(BUILD_USE_F32)
        #define M_SQRT_2_OVER_PI 0.79788456f
        #define M_SQRT_2_OVER_PI_A 0.044715f
        #elif defined(BUILD_USE_F16)
        #define M_SQRT_2_OVER_PI 0.79788456_c
        #define M_SQRT_2_OVER_PI_A 0.044715_c
        #elif defined(BUILD_USE_BF16)
        #define M_SQRT_2_OVER_PI 0.79788456bf16
        #define M_SQRT_2_OVER_PI_A 0.044715bf16
        #else
        #define M_SQRT_2_OVER_PI 0.79788456f
        #define M_SQRT_2_OVER_PI_A 0.044715f
        #endif

        cumeric_t tanh_arg = M_SQRT_2_OVER_PI * (x + M_SQRT_2_OVER_PI_A * x * x * x);
        cumeric_t tanh_val = tanhf(tanh_arg);
        cumeric_t sech_squared = 1 - tanh_val * tanh_val;
        return 0.5_c * (1.0_c + tanh_val) + (M_SQRT_2_OVER_PI * (x + 3 * M_SQRT_2_OVER_PI_A * x * x) * sech_squared) / 6.0_c;
    }

    cumeric_t relu(cumeric_t x) { return x > 0.0_c ? x : 0.0_c; }
    cumeric_t relu_derivative(cumeric_t x) { return x > 0.0_c ? 1.0_c : 0.0_c; }

    cumeric_t tanh(cumeric_t x) { return sycl::tanh(x); }
    cumeric_t tanh_derivative(cumeric_t x) { return 1.0_c - sycl::tanh(x) * sycl::tanh(x); }

    cumeric_t sigmoid(cumeric_t x) { return 1.0_c / (1.0_c + expf(-x)); }
    cumeric_t sigmoid_derivative(cumeric_t x) { return sigmoid(x) * (1.0_c - sigmoid(x)); }
    cumeric_t sigmoid_derivative_from_result(cumeric_t x) { return x * (1.0_c - x); }
    /* Parallel Functions */

    void linear(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        library::getQueue().copy(v, r, N);
    }

    void linearInPlace(cumeric_t* v, const std::size_t N)
    {
        // literally do nothing
    }

    void linear_derivative(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        // library::getQueue().copy(library::getOnes(), r, N);
        library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            r[i] = 1;
            // r[i] = v[i] > 0_c ? v[i] : 0_c;
        }).wait();
    }


    void relu(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            r[i] = v[i] > 0_c ? v[i] : 0_c;
        }).wait();
    }
    void reluInPlace(cumeric_t* v, const std::size_t N)
    {
        library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            v[i] = v[i] > 0_c ? v[i] : 0_c;
        }).wait();
    }

    void relu_derivative(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            r[i] = v[i] > 0_c ? 1_c : 0_c;
        }).wait();
    }
    void relu_derivativeInPlace(cumeric_t* v, const std::size_t N)
    {
        library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            v[i] = v[i] > 0_c ? 1_c : 0_c;
        }).wait();
    }

    void leaky_relu(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            r[i] = v[i] > 0_c ? v[i] : v[i] * leaky_relu_alpha;
        }).wait();
    }

    void leaky_relu_derivative(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            r[i] = v[i] > 0_c ? 1_c : leaky_relu_alpha;
        }).wait();
    }

    void tanh(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::tanh(library::getQueue(), N, v, r).wait();
    }
    void tanhInPlace(cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::tanh(library::getQueue(), N, v, v).wait();
    }

    void tanh_derivative(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        // oneapi::mkl::vm::tanh(library::getQueue(), N, v, r).wait();
        // library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        // {
            // const std::size_t i = idx[0];
            // r[i] = 1 - r[i] * r[i];
        // }).wait();

        auto queue = library::getQueue();
        auto e = oneapi::mkl::vm::tanh(queue, N, v, r);

        queue.parallel_for(
            sycl::range<1>(N),
            {e},
            [=](sycl::id<1> idx)
            {
                r[idx] = 1 - r[idx] * r[idx];
            });
    }
    void tanh_devivativeInPlace(cumeric_t* v, const std::size_t N)
    {
        oneapi::mkl::vm::tanh(library::getQueue(), N, v, v).wait();
        library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            v[i] = 1 - v[i] * v[i];
        }).wait();
    }

    void sigmoid(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {

        // LinearAlgebra::scale(r, v, -1.0_c, N);
        // oneapi::mkl::vm::exp(library::getQueue(), N, r, r).wait();
        library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            // r[i] = 1.f / (1.f + r[i]);
            r[i] = 1.f / (1.f + sycl::exp(-v[i]));
        }).wait();
    }
    void sigmoidInPlace(cumeric_t* v, const std::size_t N)
    {

    }

    void sigmoid_derivative(cumeric_t* r, const cumeric_t* v, const std::size_t N)
    {
        // LinearAlgebra::scale(r, v, -1.0_c, N);
        // oneapi::mkl::vm::exp(library::getQueue(), N, r, r).wait();
        cumeric_t* temp = cum::memory::allocate(N);
        cum::functions::sigmoid(temp, v, N);
        library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            r[i] = temp[i] * (1 - temp[i]);
            // r[i] = r[i] * (1  - r[i]);
        }).wait();
        cum::memory::free(temp);
    }
    void sigmoid_derivativeInPlace(cumeric_t* v, const std::size_t N)
    {

    }
    /* Functors Functions */

    cumeric_t Linear::operator()(cumeric_t x) const { return linear(x); }
    
    cumeric_t GELU::operator()(cumeric_t x) const { return gelu(x); }
     
    cumeric_t ReLU::operator()(cumeric_t x) const { return relu(x); }

    cumeric_t Tanh::operator()(cumeric_t x) const { return sycl::tanh(x); }

    cumeric_t Sigmoid::operator()(cumeric_t x) const { return sigmoid(x); }

    /* Various non x->y functions */

    void fill(cumeric_t* v, cumeric_t val, const std::size_t N)
    {
        library::getQueue().parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];
            v[i] = val;
        }).wait();
    }

    cumeric_t min(const cumeric_t a, const cumeric_t b) { return sycl::min(a, b); }
    
    cumeric_t max(const cumeric_t a, const cumeric_t b) { return sycl::max(a, b); }

    // temporary naive implementation
    void clip(cumeric_t* r, const cumeric_t* v, const cumeric_t min, const cumeric_t max, const std::size_t N)
    {
        for(size_t i = 0 ; i < N ; i++) { r[i] = v[i] > min && v[i] < max ? v[i] : v[i] < min ? min : max; };
    }

    void clipInPlace(cumeric_t* v, const cumeric_t min, const cumeric_t max, const std::size_t N)
    {
        for(size_t i = 0 ; i < N ; i++) { v[i] = v[i] > min && v[i] < max ? v[i] : v[i] < min ? min : max; };
    }

    void getFunctionByName(activation_t* activation, const char* name)
    {
        if(strcmp(name, "linear") == 0)
        {
            *activation = { ActivationFunction::linear, linear, linear_derivative };
        }
        else if(strcmp(name, "gelu") == 0)
        {
            *activation = { ActivationFunction::gelu, gelu, GELU_derivative };
        }
        else if(strcmp(name, "relu") == 0)
        {
            *activation = { ActivationFunction::relu, relu, relu_derivative };
        }
        else if(strcmp(name, "leaky_relu") == 0)
        {
            *activation = { ActivationFunction::leaky_relu, relu, relu_derivative };
        }
        else if(strcmp(name, "tanh") == 0)
        {
            *activation = { ActivationFunction::tanh, tanh, tanh_derivative };
        }
        else if(strcmp(name, "sigmoid") == 0)
        {
            *activation = { ActivationFunction::sigmoid, sigmoid, sigmoid_derivative };
        }
        else
        {
            *activation = { ActivationFunction::undefined, nullptr, nullptr };
        }
    }

    void transform(cumeric_t* r, const cumeric_t* v, activation_t& func, std::size_t N)
    {
        switch (func.name)
        {
            case ActivationFunction::linear:
            {
                // copy v to r (use queue beceuse r and v are pointing to Unified Shared Memory)
                // std::cout << "linear activation\n";
                // library::getQueue().copy(v, r, N);
                break;
            }
            case ActivationFunction::relu:
            {
                relu(r, v, N);
                break;
            }
            case ActivationFunction::leaky_relu:
            {
                leaky_relu(r, v, N);
                break;
            }
            case ActivationFunction::sigmoid:
            {
                // std::cout << "sigmoid activation\n";\
                // NAIVE L0OP

                sigmoid(r, v, N);
                // for (std::size_t i = 0; i < N; ++i)
                    // r[i] = sigmoid(v[i]);
                    // r[i] = 1.0 / (1.0 + std::exp(-v[i]));
                break;
            }
            case ActivationFunction::tanh:
            {
                tanh(r, v, N);
                break;
            }
            default:
            {
                std::cerr << "Unknown activation function" << std::endl;
            }
        }
    }

    void transformInPlace(cumeric_t* v, activation_t& func, std::size_t N)
    {
        // #pragma omp parallel for
        // for (std::size_t i = 0; i < N; ++i)
        //     v[i] = func.function(v[i]);
    }

    void activationDerivative(cumeric_t* r, const cumeric_t* v, activation_t& func, std::size_t N)
    {
        // #pragma omp parallel for
        // for (std::size_t i = 0; i < N; ++i)
        //     r[i] = func.derivative(v[i]);

        switch (func.name)
        {
            case ActivationFunction::linear:
            {
                // copy v to r (use queue beceuse r and v are pointing to Unified Shared Memory)
                linear_derivative(r, v, N);
                break;
            }
            case ActivationFunction::relu:
            {
                relu_derivative(r, v, N);
                break;
            }
            case ActivationFunction::leaky_relu:
            {
                leaky_relu_derivative(r, v, N);
                break;
            }
            case ActivationFunction::sigmoid:
            {
                for (std::size_t i = 0; i < N; ++i)
                    r[i] = (cumeric_t)sigmoid_derivative(v[i]);
                // sigmoid_derivative(r, v, N);
                break;
            }
            case ActivationFunction::tanh:
            {
                tanh_derivative(r, v, N);
                break;
            }
            default:
            {
                std::cerr << "Unknown activation function" << std::endl;
            }
        }
    }

    void activationDerivativeInPlace(cumeric_t* v, activation_t& func, std::size_t N)
    {
        #pragma omp parallel for
        for (std::size_t i = 0; i < N; ++i)
            v[i] = func.derivative(v[i]);
    }

}
