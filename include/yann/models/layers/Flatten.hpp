/**
 * @file Flatten.hpp
 * @author jaro
 * @brief Layer that flattens multidimensional input tensors into 2D matrices preserving batch dimension.
 * @date 2026-07-24
 */

#ifndef YANN_FLATTEN_HPP
#define YANN_FLATTEN_HPP

#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    /**
     * @class Flatten
     * @brief Reshapes an N-dimensional tensor [batch, d1, d2, ...] into 2D [batch, d1 * d2 * ...].
     */
    class Flatten : public LayerBase
    {
    public:
        // Flatten(int rows, int cols);

        /**
         * @brief Construct a Flatten layer with the expected input shape.
         * @param shape Input tensor shape (excluding or including dimensions).
         */
        Flatten(const cum::Shape& shape);

        /**
         * @brief Compute forward flattening pass.
         * @param input Input tensor to flatten.
         * @return 2D flattened tensor of shape [batch, total_features].
         */
        cum::Tensor forward(const cum::Tensor& input) override;

        /**
         * @brief Compute backward un-flattening pass.
         * @param deltaOutput Upstream gradient tensor of shape [batch, total_features].
         * @return Reshaped gradient matching original input shape.
         */
        cum::Tensor backward(const cum::Tensor& deltaOutput) override;

    private:
        cum::Shape shape_; /**< Recorded input shape for backward reconstruction. */
        // int input_ros;
        // int input_cols;
        // bool to_column_vector; // true by default
    };

} // yann::models::layers

#endif //YANN_FLATTEN_HPP
