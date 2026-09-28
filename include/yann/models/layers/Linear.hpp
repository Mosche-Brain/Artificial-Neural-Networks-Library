//! @file Linear.hpp
//! @brief Linear (fully connected) layer without activation specification in constructor.
//!
//! The Linear layer implements a linear transformation y = Wx + b without applying
//! an activation function. This makes it distinct from the Dense layer which requires
//! an activation function to be specified at construction time.
//!
//! @note This layer is useful when you want to compose activation functions separately
//!       or when the linear transformation is the final operation before loss computation.
//!
//! @defgroup yann_layers Linear Layer
//! @{

//! @brief Linear layer implementing y = Wx + b without activation.
//!
//! The Linear layer performs a linear transformation of the input:
//! @f[y = Wx + b@f]
//!
//! Where:
//! - @f$W@f$ is the weight matrix of shape (@f$output\_features \times input\_features@f$)
//! - @f$b@f$ is the bias vector of shape (@f$output\_features@f$)
//! - @f$x@f$ is the input vector
//!
//! Unlike Dense, the activation function is not specified at construction but can be
//! applied separately in the network architecture if needed.
//!
//! @see LayerBase
//! @see Dense (similar layer with activation specified at construction)
//! @see Input

#pragma once

#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    class Linear : public LayerBase
    {
    public:
        //! @brief Construct a Linear layer with the specified size.
        //!
        //! @param layerSize Number of neurons (output features) in this layer.
        //!
        //! @note The activation function is not set during construction. Use activation()
        //!       setter or apply activation separately in your network architecture.
        Linear(int layerSize);

        //! @brief Initialize the layer's parameters (weights and biases).
        //!
        //! @param output_features Number of output features (should match layerSize from constructor).
        //! @param input_features Number of input features (determines weight matrix width).
        //!
        //! @note Weights are initialized using uniform distribution in range [-0.1, 0.1].
        //!       Biases are initialized to zeros. The weights matrix has dimensions
        //!       (output_features, input_features), and biases have dimensions (output_features, 1).
        void init_parameters(int output_features, int input_features) override;

        //! @brief Compute the forward pass (linear transformation only).
        //!
        //! @param input Input tensor of shape (batch_size, input_features).
        //!              The last dimension must match the weight matrix's column count.
        //! @return Output tensor of shape (batch_size, output_features)
        //!         resulting from the linear transformation Wx + b.
        //!
        //! @note The input is stored in the internal cache for use during backward propagation.
        //!       No activation function is applied to the output.
        //! @see forward() for the corresponding forward pass.
        cum::Tensor forward(const cum::Tensor& input) override;

        //! @brief Compute the backward pass (linear backpropagation).
        //!
        //! @param deltaOutput Gradient of the loss with respect to the layer's output,
        //!                    of shape (batch_size, output_features).
        //! @return Gradient tensor with respect to the layer's input,
        //!         of shape (batch_size, input_features).
        //!
        //! @note During backpropagation, gradients are computed for weights, biases, and input
        //!       without any activation function derivative modulation.
        //! @see backward() for the corresponding backward pass.
        cum::Tensor backward(const cum::Tensor& deltaOutput) override;

        //! @brief Collect layer parameters for optimization.
        //!
        //! @param params Vector of Parameter pointers to which this layer's parameters will be added.
        //!               The vector will have two entries added: weights and biases.
        //!
        //! @note The weights and biases must be initialized (via init_parameters) before calling
        //!       this method.
        void collect_parameters(std::vector<Parameter*>& params) override;

        //! @brief Create a Linear layer using unique pointer.
        //!
        //! @param layerSize Number of neurons (output features) in the layer.
        //! @return Owned pointer to the created Linear layer.
        static std::unique_ptr<LayerBase> createUnique(int layerSize);

        //! @brief Get the weights matrix (override from LayerBase).
        //!
        //! @return Reference to the weight tensor values.
        cum::Tensor& weights() override { return weights_.values; }

        //! @brief Get the weights gradient.
        //!
        //! @return Reference to the weight gradient tensor.
        cum::Tensor& weights_grad() override { return weights_.gradient; }

        //! @brief Get the biases vector (override from LayerBase).
        //!
        //! @return Reference to the bias tensor values.
        cum::Tensor& biases() override { return biases_.values; }

        //! @brief Get the biases gradient.
        //!
        //! @return Reference to the bias gradient tensor.
        cum::Tensor& biases_grad() override { return biases_.gradient; }
    private:
        Parameter weights_;
        Parameter biases_;
    };
}
//! @}
