/**
 * @file Normalize.hpp
 * @author jaro
 * @brief Normalization layer implementation.
 * @date 2026-09-26
 */

#pragma once

#include <memory>
#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    /**
     * @class Normalize
     * @brief Normalization layer standardizing feature activations.
     */
    class Normalize : public LayerBase
    {
    public:
        /**
         * @brief Construct a Normalize layer for 1D features.
         * @param lenght Number of features to normalize.
         */
        Normalize(cum::dim_t lenght);

        /**
         * @brief Construct a Normalize layer for arbitrary shape.
         * @param shape Shape of the input tensor.
         */
        Normalize(const cum::Shape& shape);

        /**
         * @brief Perform forward normalization pass.
         * @param input Input tensor.
         * @return Normalized output tensor.
         */
        cum::Tensor forward(const cum::Tensor& input) override;

        /**
         * @brief Perform backward normalization pass.
         * @param grad Upstream gradient tensor.
         * @return Gradient with respect to input.
         */
        cum::Tensor backward(const cum::Tensor& grad) override;

        /**
         * @brief Factory method creating a unique pointer to a Normalize layer.
         * @param shape Input tensor shape.
         * @return std::unique_ptr<LayerBase> Newly created layer.
         */
        std::unique_ptr<LayerBase> createUnique(const cum::Shape& shape);

    private:
    };
} // yann::models::layers
