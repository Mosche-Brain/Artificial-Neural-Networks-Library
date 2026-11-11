/**
 * @file Softmax.hpp
 * @author jaro
 * @brief Softmax activation layer computing normalized probability distributions.
 * @date 2026-10-02
 */

#pragma once

#include <memory>
#include <vector>

#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    struct SoftmaxCache : LayerCache
    {
        cum::Tensor a;
        cum::Tensor da;
    };

    /**
     * @class Softmax
     * @brief Computes the softmax activation over an input tensor along a specified axis:
     * @f[
     * \text{Softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
     * @f]
     */
    class Softmax : public LayerBase
    {
    public:
        using Cache = SoftmaxCache;

        /**
         * @brief Construct a Softmax layer with feature length and target axis.
         * @param lenght Dimension length along feature axis.
         * @param axis Axis along which softmax is evaluated (default: -1, the last axis).
         */
        Softmax(cum::dim_t lenght, cum::dim_t axis = -1);

        /**
         * @brief Construct a Softmax layer with shape and target axis.
         * @param shape Shape of the input tensor.
         * @param axis Axis along which softmax is evaluated (default: -1).
         */
        Softmax(const cum::Shape& shape, cum::dim_t axis = -1);

        /**
         * @brief Initialize layer parameters (no-op as Softmax has no trainable weights).
         * @param output_features Output feature dimension.
         * @param input_features Input feature dimension.
         */
        void init_parameters(int output_features, int input_features) override;

        /**
         * @brief Initialize layer parameters given tensor shapes.
         * @param input_shape Input tensor shape.
         * @param output_shape Output tensor shape.
         */
        void init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape) override;

        /**
         * @brief Forward pass computing softmax probabilities.
         * @param input Input tensor of logits.
         * @return Normalized probability distribution tensor.
         */
        cum::Tensor forward(const cum::Tensor& input) override;

        /**
         * @brief Backward pass propagating gradients through softmax Jacobian.
         * @param gradient Upstream gradient tensor.
         * @return Gradient with respect to input logits.
         */
        cum::Tensor backward(const cum::Tensor& gradient) override;

        /**
         * @brief Collect parameters (no-op for Softmax).
         * @param params Parameter vector (unchanged).
         */
        void collect_parameters(std::vector<Parameter*>& params) override;

        /**
         * @brief Factory method creating a unique pointer to a Softmax layer.
         * @param lenght Feature length.
         * @param axis Reduction axis (default: -1).
         * @return std::unique_ptr<LayerBase> Newly created Softmax layer.
         */
        static std::unique_ptr<LayerBase> createUnique(cum::dim_t lenght, cum::dim_t axis = -1);

        /**
         * @brief Factory method creating a unique pointer to a Softmax layer from shape.
         * @param shape Tensor shape.
         * @param axis Reduction axis (default: -1).
         * @return std::unique_ptr<LayerBase> Newly created Softmax layer.
         */
        static std::unique_ptr<LayerBase> createUnique(const cum::Shape& shape, cum::dim_t axis = -1);

    private:
        cum::dim_t axis_;   /**< Normalization axis index. */
        Cache cache;        /**< Internal cache for forward activations during backward propagation. */
    };
} // yann::models::layers
