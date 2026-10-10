//! @file Input.hpp
//! @brief Input layer that serves as the entry point for data into a neural network.
//!
//! The Input layer is a special layer that doesn't learn parameters but instead
//! provides the input tensor to the rest of the network. It acts as the first
//! layer in a Sequential model, defining the expected input shape.
//!
//! @note The Input layer does not have trainable weights or biases. Its purpose
//!       is to accept input data and pass it forward while recording the input
//!       shape for subsequent layers.
//!
//! @defgroup yann_layers Input Layer
//! @{

//! @brief Input layer that serves as the data entry point for neural networks.
//!
//! The Input layer does not contain trainable parameters. It simply accepts an
//! input tensor and passes it forward to the next layer in the topology. The
//! input shape is recorded and can be queried via input_shape() and output_shape().
//!
//! @note This layer is typically not used directly in code; instead, the first
//!       Dense or Linear layer's input shape is inferred from the data. However,
//!! it can be explicitly added to specify the expected input dimensionality.
//!
//! @see LayerBase
//! @see Dense
//! @see Linear

#pragma once

#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    class Input : public LayerBase
    {
    public:
        //! @brief Construct an Input layer with the specified size.
        //!
        //! @param layerSize Number of input features (dimension of the input data).
        //!
        //! @note The input layer creates an identity mapping: the output equals the input.
        //!       The layerSize parameter defines the expected input dimensionality.
        Input(int layerSize);

        //! @brief Initialize parameters (no-op for Input layer).
        //!
        //! @param output_features Not used for Input layer (maintained for interface compatibility).
        //! @param input_features Not used for Input layer (maintained for interface compatibility).
        //!
        //! @note This method overrides the pure virtual function from LayerBase but does
        //!       nothing since Input layers don't have trainable parameters.
        void init_parameters(int output_features, int input_features) override;

        //! @brief Compute the forward pass (identity mapping).
        //!
        //! @param input Input tensor of shape (batch_size, input_features).
        //! @return The same input tensor, unchanged.
        //!
        //! @note The input is stored in the internal cache for backward pass compatibility.
        //!       The output is a direct pass-through of the input.
        cum::Tensor forward(const cum::Tensor& input) override;

        //! @brief Compute the backward pass (pass gradient to input).
        //!
        //! @param deltaOutput Gradient from the next layer, of shape (batch_size, output_features).
        //! @return The same gradient tensor, passed through to the previous layer.
        //!
        //! @note For an Input layer, the gradient is simply passed through unchanged,
        //!       since there are no parameters to update and the forward was an identity.
        cum::Tensor backward(const cum::Tensor& deltaOutput) override;

        //! @brief Collect parameters (no-op for Input layer).
        //!
        //! @param params Vector to which parameters would be added (none added for Input).
        //!
        //! @note This method overrides the pure virtual function from LayerBase but does
        //!       nothing since Input layers don't have trainable parameters. The params
        //!       vector remains unchanged.
        void collect_parameters(std::vector<Parameter*>& params) override;

        //! @brief Create an Input layer using unique pointer.
        //!
        //! @param layerSize Number of input features for the layer.
        //! @return Unique pointer to the created Input layer.
        static std::unique_ptr<LayerBase> createUnique(int layerSize);
    };
}
//! @}
