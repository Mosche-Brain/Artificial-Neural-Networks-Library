//
// Created by jaro on 8/31/26.
//

#include "yann/logging/Logger.hpp"
#include "yann/runtime_config.hpp"

#include "yann/models/layers/Linear.hpp"

#define ENABLE_RUNTIME_CHECKS true

namespace yann::models::layers
{
    Linear::Cache::LinearCache(int input_features, int output_features, cum::datatype dtype)
    {
        this->z       = cum::Tensor(cum::Shape{output_features}, dtype);
        this->dz      = cum::Tensor(cum::Shape{output_features}, dtype);

        owned = { &z, &dz };
    }


    Linear::Linear(const int layerSize)
    {
        YANN_LOG(1, "Initializing Linear layer with {} neurons...", layerSize);
        output_shape_ = { layerSize };

        _kind = Kind::Linear;
    }

    void Linear::init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape)
    {
        this->input_shape_ = input_shape;
        this->output_shape_ = output_shape;
        cum::dim_t input_features = input_shape.empty() ? 1 : input_shape.back();
        cum::dim_t output_features = output_shape.empty() ? 1 : output_shape.back();
        init_parameters(static_cast<int>(output_features), static_cast<int>(input_features));
    }

    void Linear::init_parameters(int output_features, int input_features)
    {
        this->input_shape_  = cum::Shape{input_features};
        this->output_shape_ = cum::Shape{output_features};
        this->weights_      = Parameter::Uniform(input_features, output_features);   /* input_length * neurons */
        this->biases_       = Parameter::Zeros(1, output_features);               /* Row-Vector */

        this->cache = Cache(input_features, output_features);

        this->_initialized = true;
    }

    Tensor Linear::forward(const Tensor& input)
    {
        constexpr bool YANN_LINEAR_FORWARD_FUSED_PATH = true;
        constexpr bool YANN_LINEAR_FORWARD_REFERENCE_PATH = true;

        // check if number of weights rows is equal to number of input features (input is [batch, features])
        if constexpr(ENABLE_RUNTIME_CHECKS)
        {
            if(input.cols() != weights_.rows())
            {
                throw std::runtime_error("Input dimension mismatch: " + std::to_string(input.cols()) + " != " + std::to_string(weights_.rows()));
            }
        }

        cum::dim_t batch = input.shape().at(0);

        cache.x = input;
        cache.z = Tensor({batch, weights_.cols()}, cache.z.type());

        if(runtime_config::fused_kernels())
        {
            if constexpr(YANN_LINEAR_FORWARD_FUSED_PATH)
            {
                cache.matmul_pd = Primitive::Descriptor::Matmul(
                    *cache.z.descriptor(), *cache.x.descriptor(), *weights_.values.descriptor(), *biases_.values.descriptor()
                );

                const auto matmul = Primitive(cache.matmul_pd);

                YANN_LOG(3, "cache.z = input * weights_.values + biases_.values", "");
                YANN_LOG(4, "X: {}x{}, | W: {}x{} | B: {}x{}", input.rows(), input.cols(), weights_.values.rows(), weights_.values.cols(), biases_.values.rows(), biases_.values.cols());
                matmul.execute({
                    { Primitive::Argument::dst, *cache.z.memory() },
                    { Primitive::Argument::src, *cache.x.memory() },
                    { Primitive::Argument::weights, *weights_.values.memory() },
                    { Primitive::Argument::bias, *biases_.values.memory() } ,
                }).wait();

                return cache.z;
            }
            else
            {
                throw std::runtime_error("YANN_LINEAR_FORWARD_FUSED_PATH weren't compiled");
            }
        }
        else
        {
            if constexpr(YANN_LINEAR_FORWARD_REFERENCE_PATH)
            {

                YANN_LOG(3, "cache.a = input * weights_.values", "");
                YANN_LOG(4, "X: {}x{}, | W: {}x{}", input.rows(), input.cols(), weights_.values.rows(), weights_.values.cols());
                cache.z = input * weights_.values;


                YANN_LOG(3, "cache.a = cache.z + biases_.values", "");
                YANN_LOG(4, "A: {}x{} | B: {}x{}", cache.z.rows(), cache.z.cols(), biases_.values.rows(), biases_.values.cols());
                cache.z = cache.z + biases_.values; // with broadcast

                return cache.z;
            }
            else
            {
                throw std::runtime_error("YANN_LINEAR_FORWARD_REFERENCE_PATH weren't compiled");
            }
        }
    }

    cum::Tensor Linear::backward(const cum::Tensor& deltaOutput)
    {
        constexpr bool YANN_LINEAR_BACKWARD_FUSED_PATH = false;
        constexpr bool YANN_LINEAR_BACKWARD_REFERENCE_PATH = true;

        if(runtime_config::fused_kernels())
        {
            if constexpr(YANN_LINEAR_BACKWARD_FUSED_PATH)
            {
                throw std::logic_error("YANN_LINEAR_BACKWARD_FUSED_PATH not implemented");
            }
            else
            {
                throw std::runtime_error("YANN_LINEAR_BACKWARD_FUSED_PATH weren't compiled");
            }
        }
        else
        {
            if constexpr(YANN_LINEAR_BACKWARD_REFERENCE_PATH)
            {
                YANN_LOG(4, "dL/dZ = deltaOutput * ones", "");
                cache.dz = deltaOutput;

                YANN_LOG(4, "dL/dB = sum of cache.dz over batch axis", "");
                biases_.gradient += cache.dz.sum(0, true);

                YANN_LOG(4, "dL/dW = cache.x^T * cache.dz", "");
                weights_.gradient += cache.x.transpose() * cache.dz;

                return cache.dz * weights_.values.transpose();
            }
            else
            {
                throw std::runtime_error("YANN_LINEAR_FORWARD_REFERENCE_PATH weren't compiled");
            }
        }
    }

    void Linear::collect_parameters(std::vector<Parameter*>& params)
    {
        params.push_back(&weights_);
        params.push_back(&biases_);
    }

    std::unique_ptr<LayerBase> Linear::createUnique(int layerSize)
    {
        return std::make_unique<Linear>(layerSize);
    }
} // yann