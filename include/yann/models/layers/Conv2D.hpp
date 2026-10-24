/**
 * @file Conv2D.hpp
 * @author jaro
 * @brief 2D spatial convolution layer (NCHW layout) with configurable kernels, stride, padding, and dilation.
 * @date 2026-08-31
 */

#ifndef YANN_CONV2D_HPP
#define YANN_CONV2D_HPP

#include <vector>
#include <memory>

#include "cum/Core.hpp"
#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    /**
     * @class Conv2D
     * @brief Two-dimensional spatial convolutional layer assuming NCHW tensor layout.
     *
     * Computes 2D cross-correlation over spatial feature maps using learned convolution kernels.
     */
    class Conv2D : public LayerBase // NCHW layout
    {
    public:
        /**
         * @brief Construct a Conv2D layer.
         * @param input_shape Shape of the input tensor [C, H, W].
         * @param kernels_n Number of output convolution kernels / feature maps.
         * @param kernel_size Spatial dimensions of the kernel [Kh, Kw].
         * @param activation Name of the activation function to apply.
         * @param stride Stride step size along spatial dimensions (default: 1).
         * @param padding Spatial zero-padding size (default: 1).
         * @param dilatation Dilation factor for atrous convolution (default: 1).
         */
        Conv2D(const cum::Shape& input_shape, cum::dim_t kernels_n, const cum::Shape& kernel_size, const char* activation, cum::dim_t stride=1, cum::dim_t padding=1, cum::dim_t dilatation=1);

        /**
         * @brief Destructor.
         */
        ~Conv2D();

        /**
         * @brief Parameter initialization override (no-op as Conv2D initializes parameters in constructor/shape).
         * @param output_features Number of output features.
         * @param input_features Number of input features.
         */
        void init_parameters(int output_features, int input_features) override {};

        /**
         * @brief Compute forward convolution pass.
         * @param input Input tensor in NCHW layout.
         * @return Output feature maps tensor in NCHW layout.
         */
        cum::Tensor forward(const cum::Tensor& input) override;

        /**
         * @brief Compute backward convolution pass.
         * @param input Gradient flowing backwards from the next layer.
         * @return Gradient tensor with respect to input.
         */
        cum::Tensor backward(const cum::Tensor& input) override;

        // cum::Matrix forward(const cum::Matrix& input) override {}
        // cum::Matrix backward(const cum::Matrix& input) override {}

        /**
         * @brief Collect trainable kernel parameters.
         * @param params Vector to append Parameter pointers into.
         */
        void collect_parameters(std::vector<Parameter*>& params) override {};

        /**
         * @brief Factory method creating a unique instance of Conv2D.
         * @param input_shape Shape of input [C, H, W].
         * @param kernels_n Number of output kernels.
         * @param kernel_size Kernel dimensions [Kh, Kw].
         * @param activation Activation function name.
         * @param stride Stride along spatial dimensions (default: 1).
         * @param padding Padding along spatial dimensions (default: 1).
         * @param dilatation Dilation rate (default: 1).
         * @return std::unique_ptr<LayerBase> Newly created Conv2D layer.
         */
        std::unique_ptr<LayerBase> createUnique(cum::Shape input_shape, cum::dim_t kernels_n, cum::Shape kernel_size, const char* activation, cum::dim_t stride=1, cum::dim_t padding=1, cum::dim_t dilatation=1);

    private:
        struct Config 
        { 
            cum::dim_t kernels_n;     /**< Number of output kernels. */
            cum::Shape kernel_size;   /**< Kernel spatial dimensions. */
            cum::dim_t stride;        /**< Stride step size. */
            cum::dim_t padding;       /**< Zero-padding size. */
            cum::dim_t dilatation;    /**< Dilation factor. */
        } config;

        Parameter kernels;
        // std::vector<Parameter> kernels;
    };
} // yann::models::layers

#endif //YANN_CONV2D_HPP
