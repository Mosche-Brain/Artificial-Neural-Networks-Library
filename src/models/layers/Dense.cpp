#include <stdexcept>
#include <print>
#include <string_view>

#include <cum/memory.hpp>
#include <cum/runtime.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/functions.hpp>
#include <cum/functions/transform.hpp>
#include <cum/neural_primitives/neural_kernels.hpp>
#include <cum/neural_primitives/elementwise.hpp>
#include <cum/neural_primitives/elementwise_diffs.hpp>

#include "yann/runtime_config.hpp"

#include "Dense.hpp"

#include <cum/neural_primitives/Primitive.hpp>

#define ENABLE_RUNTIME_CHECKS true
#define USE_FUSED_KERNELS false
#define ENABLE_CACHED_PREACTIVATION true

/* TODO:
 * Ograniczenie kopiowania, cache.x może przechywać referencje lub view na cache.a z poprzedniej warstwy
 * Wprowadzenie forward trace selection do runtime config oraz pozostawienie branchy w `if constexpr` by za pomocą konfiguracji cmake można było wyłączyć ich kompilacje
 * Ścieżka z fused kernel dla backward
 * ~~Usunięcie `cum::runtime::sync() pozostałych po debugowaniu~~ (done?)
 * Zwracanie referencji lub widoku do cache przez `forward` i `backward` zamiast kopii
 * Brak niejawnej alokacji cache potrzebnego tylko do treningu
 */

namespace cum::dl { using namespace cum::neural_primitives; }

namespace yann::models::layers
{
    Dense::Cache::DenseCache(int input_features, int output_features, cum::datatype dtype)
    {
        this->z       = cum::Tensor(cum::Shape{output_features}, dtype);
        this->a       = cum::Tensor(cum::Shape{output_features}, dtype);
        this->dz      = cum::Tensor(cum::Shape{output_features}, dtype);
        this->da      = cum::Tensor(cum::Shape{output_features}, dtype);

        owned = { &z, &a, &da, &dz };
    }

    Dense::Dense(const int size, const Activation& activation) : _activation(activation)
    {
        YANN_LOG(1, "Initializing Dense layer with {} neurons and {} activation function...", size, activation.label);

        output_shape_ = { size };

    }

    void Dense::init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape)
    {
        this->input_shape_ = input_shape;
        this->output_shape_ = output_shape;
        cum::dim_t input_features = input_shape.empty() ? 1 : input_shape.back();
        cum::dim_t output_features = output_shape.empty() ? 1 : output_shape.back();
        init_parameters(static_cast<int>(output_features), static_cast<int>(input_features));
    }

    void Dense::init_parameters(int output_features, int input_features)
    {
        this->input_shape_  = cum::Shape{input_features};
        this->output_shape_ = cum::Shape{output_features};
        this->weights_      = Parameter::Uniform(input_features, output_features);   /* input_length * neurons */
        this->biases_       = Parameter::Zeros(1, output_features);                  /* Row-Vector */

        this->cache = Cache(input_features, output_features);

        this->_initialized = true;
    }

    
    Tensor Dense::forward(const Tensor& input) // temporary implementation for compatibility
    {
        // those const expresions will be moved to build config
        constexpr bool YANN_DENSE_FORWARD_FUSED_PATH = true;
        constexpr bool YANN_DENSE_FORWARD_REFERENCE_PATH = true;

        // check if number of weights rows is equal to number of input features (input is [batch, features])
        if constexpr(ENABLE_RUNTIME_CHECKS)
        {
            if(input.cols() != weights_.rows())
            {
                throw std::runtime_error("Input dimension mismatch: " + std::to_string(input.cols()) + " != " + std::to_string(weights_.rows()));
            }
        }

        cache.x = input;

        /* We can cache most common shapes in future */
        cache.z = Tensor(input.rows(), weights_.cols());
        cache.a = Tensor(input.rows(), weights_.cols());
        cache.da = Tensor(input.rows(), weights_.cols());

        if(runtime_config::fused_kernels())
        {
            if constexpr(YANN_DENSE_FORWARD_FUSED_PATH)
            {
                cache.matmul_pd = Primitive::Descriptor::Matmul(
                    *cache.z.descriptor(), *cache.x.descriptor(), *weights_.values.descriptor(), *biases_.values.descriptor()
                );

                cache.unary_pd = Primitive::Descriptor::Elementwise(
                    _activation.fwd, *cache.a.descriptor(), *cache.z.descriptor(), _mode == Mode::training ? cum::dl::Purpose::forward_training : cum::dl::Purpose::forward_inference
                );

                const auto matmul = Primitive(cache.matmul_pd);
                const auto unary  = Primitive(cache.unary_pd);

                matmul.execute({
                    { Primitive::Argument::dst, *cache.z.memory() },
                    { Primitive::Argument::src, *cache.x.memory() },
                    { Primitive::Argument::weights, *weights_.values.memory() },
                    { Primitive::Argument::bias, *biases_.values.memory() } ,
                }).wait();

                YANN_LOG(1, "{} activation", _activation.label);
                unary.execute({
                  { Primitive::Argument::dst, *cache.a.memory() },
                  { Primitive::Argument::src, *cache.z.memory() }
                }).wait();

                return cache.a;
            }
            else    
            {
                throw std::runtime_error("YANN_DENSE_FORWARD_FUSED_PATH weren't compiled");
            }
        }
        else
        {
            if constexpr(YANN_DENSE_FORWARD_REFERENCE_PATH)
            {
                YANN_LOG(3, "cache.z = input * weights_.values | X: {}x{}, | W: {}x{}", input.rows(), input.cols(), weights_.values.rows(), weights_.values.cols());
                // cache.z = input * weights_.values;
                cache.z = input.matmul(weights_.values);

                YANN_LOG(3, "cache.z = cache.z + biases_.values | Z: {}x{} | B: {}x{}", cache.z.rows(), cache.z.cols(), biases_.values.rows(), biases_.values.cols());
                cache.z = cache.z + biases_.values; // with broadcast

                cache.a = cache.z.elementwise(activation.name);
                YANN_LOG(3, "forward completed", "");

                return cache.a;
            }
            else
            {
                throw std::runtime_error("YANN_DENSE_FORWARD_REFERENCE_PATH weren't compiled");
            }
        }
    }

    Tensor Dense::backward(const Tensor& dL_dY)
    {
        constexpr bool YANN_DENSE_BACKWARD_FUSED_PATH = false;
        constexpr bool YANN_DENSE_BACKWARD_REFERENCE_PATH = true;

        // if(false)
        // {
        //     if constexpr(YANN_DENSE_BACKWARD_FUSED_PATH)
        //     {
        //         // Primitive::Descriptor d
        //         // throw std::logic_error("YANN_DENSE_BACKWARD_FUSED_PATH not implemented");
        //     }
        //     else
        //     {
        //         goto reference_path; // silent fallback
        //         // throw std::runtime_error("YANN_DENSE_BACKWARD_FUSED_PATH weren't compiled");
        //     }
        // }
        // else // ta ścieżka jest używana
        {
            reference_path:
            if constexpr(YANN_DENSE_BACKWARD_REFERENCE_PATH) // We may move transpositions to compute kernel, but for now it's fine
            {
                YANN_LOG(3, "dA/dZ = cache.z.elementwise_diff(activation.name)", "");
                std::println("sex sex sex");
                // cache.da = cache.z.elementwise_diff(activation.name);
                const auto da_dz_pd = Primitive::Descriptor::ElementwiseBackward(
                    _activation.bwd, *cache.a.descriptor(), *cache.da.descriptor(), *cache.z.descriptor(), cache.unary_pd
                );

                Tensor ones = Tensor::Ones(cache.z.shape());

                Primitive(da_dz_pd).execute({
                    { Primitive::Argument::diff_dst, *ones.memory() },
                    { Primitive::Argument::diff_src, *cache.da.memory() },
                    { Primitive::Argument::src, *cache.z.memory() },
                }).wait();

                YANN_LOG(1, "da mean {}", cache.da.mean());

                YANN_LOG(3, "dL/dZ = deltaOutput * cache.da", "");
                cache.dz = cache.da.cwiseProduct(dL_dY);

                YANN_LOG(3, "dL/dB = sum of cache.dz over batch axis", "");
                biases_.gradient += cache.dz.sum(0, true);

                YANN_LOG(3, "dL/dW = cache.x^T * cache.dz | dZ: {}x{} | X: {}x{}", cache.dz.rows(), cache.dz.cols(), cache.x.rows(), cache.x.cols());
                weights_.gradient += cache.x.transpose().matmul(cache.dz);

                YANN_LOG(3, "dL/dX = cache.dz * weights^T | W: {}x{}, | dZ: {}x{}", weights_.values.rows(), weights_.values.cols(), cache.dz.rows(), cache.dz.cols());
                return cache.dz * weights_.values.transpose();
            }
            else
            {
                throw std::runtime_error("YANN_DENSE_BACKWARD_REFERENCE_PATH weren't compiled");
            }
        }
    }

    void Dense::collect_parameters(std::vector<Parameter*>& params)
    {
        params.push_back(&weights_);
        params.push_back(&biases_);
    }

    std::unique_ptr<LayerBase> Dense::createUnique(int layerSize, const Activation& activation)
    {
        return std::make_unique<Dense>(layerSize, activation);
    }
}
