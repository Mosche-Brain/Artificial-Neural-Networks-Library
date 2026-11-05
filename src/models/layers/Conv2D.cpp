//
// Created by jaro on 8/31/26.
//

#include <stdexcept>
#include <ranges>

#include <cum/neural_primitives/Primitive.hpp>
#include <cum/neural_primitives/Memory.hpp>

#include "yann/logging/Logger.hpp"

#include "yann/models/layers/Conv2D.hpp"

namespace cum::nn { using namespace cum::neural_primitives; }

namespace yann::models::layers
{
    namespace
    {
        cum::Shape extend_shape(const cum::Shape& shape, cum::dim_t batch=1)
        {
            cum::Shape new_shape = shape;
            new_shape.insert(new_shape.begin(), batch);

            return new_shape;
        }
    }

    Conv2D::Conv2D(const cum::Shape& input_shape, const cum::dim_t kernels_n, const cum::Shape& kernel_size, const cum::dim_t stride, const cum::dim_t padding, const cum::dim_t dilatation)
    {
        this->input_shape_ = input_shape;
        this->_layerType_ = LayerType::Conv2D;
        config = {
            .kernels_n=kernels_n,
            .kernel_size=kernel_size,
            .stride=stride,
            .padding=padding,
            .dilatation=dilatation
        };

        cum::dim_t h_out = input_shape_[1] - kernels_n + 2 * padding;
        cum::dim_t w_out = input_shape_[2] - kernels_n + 2 * padding;
        output_shape_ = { config.kernels_n, h_out, w_out};

        init_parameters(input_shape_, output_shape_);

        YANN_LOG(1, "Initialized Conv2D layer with {} input channels, {} output channels, {} height and {} width", input_shape[0], kernels_n, input_shape[1], input_shape[2]);
    }

    Conv2D::~Conv2D()
    {

    }

    // void Conv2D::init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape)
    void Conv2D::init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape)
    {
        this->input_shape_ = input_shape;
        this->output_shape_ = output_shape;

        kernels = Parameter::Uniform({
            config.kernels_n,
            input_shape[1],
            config.kernel_size[0],
            config.kernel_size[1],
        }, cum::layout::OIHW);

        biases = Parameter::Uniform({
            config.kernels_n
        });

        this->_initialized_ = true;
    }

    cum::Tensor Conv2D::forward(const cum::Tensor& input)
    {
        // constexpr bool YANN_CONV2D_BACKWARD_FUSED_PATH = true;
        // constexpr bool YANN_CONV2D_BACKWARD_REFERENCE_PATH = true;

        cum::Shape shape;
        if (input.dims() == 3)
            shape = extend_shape(input_shape_);
        else if (input.dims() == 4)
            shape = input_shape_;
        else
            throw std::runtime_error("Conv2D::forward: invalid input tensor shape");

        cache.x = input;

        cum::nn::Memory::Descriptor in_md(shape, input.type(), input.format());
        cum::nn::Memory::Descriptor out_md(extend_shape(output_shape_, shape[0]), input.type(), input.format());

        cache.z = cum::Tensor(out_md);

        auto pd = cum::nn::PrimitiveDescriptor::Conv2D(
            cum::nn::Algo::convolution,
            out_md, in_md,
            *kernels.values.descriptor(), *biases.values.descriptor(),
            {config.stride}, {config.padding},
            cum::nn::Purpose::forward_inference
        );

        cum::nn::Primitive primitive(pd);
        primitive.execute({
            {cum::nn::Primitive::Argument::src, *cache.x.memory()},
            {cum::nn::Primitive::Argument::weights, *kernels.values.memory()},
            {cum::nn::Primitive::Argument::bias, *biases.values.memory()},
            {cum::nn::Primitive::Argument::dst, *cache.z.memory()}
        });

        return cache.z;
    }

    cum::Tensor Conv2D::backward(const cum::Tensor& input)
    {
        throw std::logic_error("Conv2D::backward not yet implemented");
    }

    std::unique_ptr<LayerBase> Conv2D::createUnique(const cum::Shape input_shape, const cum::dim_t kernels_n, const cum::Shape kernel_size, const cum::dim_t stride, const cum::dim_t padding, const cum::dim_t dilatation)
    {
        return std::make_unique<Conv2D>(input_shape, kernels_n, kernel_size, stride, padding, dilatation);
    }
  
} // yann