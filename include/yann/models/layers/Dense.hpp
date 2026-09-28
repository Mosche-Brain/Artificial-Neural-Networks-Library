//! @file Dense.hpp
//! @brief Fully connected (dense) neural network layer implementation.
//!
//! The Dense layer implements a fully connected layer with configurable activation function.
//! It maintains weight and bias parameters and performs forward and backward passes
//! using the cum computational backend.
//!
//! @note This layer expects 2D tensor inputs with shape (batch_size, input_features).
//!       The weights matrix has dimensions (output_features, input_features).

//! @defgroup yann_layers Dense Layer
//! @{

//! @brief Fully connected (dense) neural network layer.
//! @details The Dense layer implements a linear transformation followed by an activation function:
//! @f[y = \sigma(Wx + b)@]
//!
//! Where:
//! - @f$W@f$ is the weight matrix of shape (@f$output\_features \times input\_features@f$)
//! - @f$b@f$ is the bias vector of shape (@f$output\_features@f$)
//! - @f$x@f$ is the input vector
//! - @f$\sigma@f$ is the activation function
//!
//! The layer maintains internal cache for storing intermediate values needed during
//! the backward pass (pre-activation values @f$z@f$ and post-activation values @f$a@f$).

#pragma once

#include "LayerBase.hpp"

namespace yann::models::layers
{
    //! @brief Fully connected (dense) neural network layer.
    //!
    //! The Dense layer implements a linear transformation followed by an activation function:
    //! @f[y = \sigma(Wx + b)@]
    //!
    //! @see LayerBase
    //!
    class Dense : public LayerBase
    {
    public:
        //! @brief Construct a Dense layer with the specified size and activation function.
        //!
        //! @param layerSize Number of neurons (output features) in this layer.
        //! @param func Name of the activation function to use (e.g., "relu", "sigmoid", "tanh").
        //!             The function must be registered with cum::functions::get_function_by_name.
        //!
        //! @note The activation function is looked up at construction time and stored for
        //!       use in forward/backward passes. Invalid function names will cause runtime errors
        //!       during forward computation.
        Dense(int layerSize, const char* func);

        //! @brief Initialize the layer's parameters (weights and biases).
        //!
        //! Weights are initialized using uniform distribution in range [-0.1, 0.1].
        //! Biases are initialized to zeros.
        //!
        //! @param output_features Number of output features (should match layerSize from constructor).
        //! @param input_features Number of input features (determines weight matrix width).
        //!
        //! @note This method overrides the pure virtual function from LayerBase.
        //!       The weights matrix has dimensions (output_features, input_features),
        //!       and biases have dimensions (output_features, 1).
        void init_parameters(int output_features, int input_features) override;

        //! @brief Compute the forward pass of the dense layer.
        //!
        //! Performs the transformation: @f$z = Wx + b@f$ followed by activation: @f$a = \sigma(z)@f$.
        //!
        //! @param input Input tensor of shape (@f$batch\_size \times input\_features@f$).
        //!              The last dimension must match the weight matrix's column count (@f$input\_features@f$).
        //! @return Output tensor of shape (@f$batch\_size \times output\_features@f$),
        //!         after applying the activation function.
        //!
        //! @note The input is stored in the internal cache for use during backward propagation.
        //!       The output activations are also cached for potential use in subsequent operations.
        //!
        //! @see backward() for the corresponding backward pass.
        cum::Tensor forward(const cum::Tensor& input) override;

        //! @brief Compute the backward pass of the dense layer.
        //!
        //! Calculates gradients of the loss with respect to weights, biases, and input.
        //!
        //! @param deltaOutput Gradient of the loss with respect to the layer's output,
        //!                    of shape (@f$batch\_size \times output\_features@f$).
        //! @return Gradient tensor with respect to the layer's input,
        //!         of shape (@f$batch\_size \times input\_features@f$).
        //!
        //! @note During backpropagation, the following gradients are computed and cached:
        //!       - @f$\partial L / \partial W@f$ (weights gradient)
        //!       - @f$\partial L / \partial b@f$ (biases gradient)
        //!       - @f$\partial L / \partial x@f$ (input gradient for previous layer)
        //!
        //! @see forward() for the corresponding forward pass.
        cum::Tensor backward(const cum::Tensor& deltaOutput) override;

        //! @brief Collect layer parameters for optimization.
        //!
        //! Adds the layer's weights and biases to the provided parameter vector.
        //! These parameters can then be used by optimizers (SGD, Adam, etc.) for update steps.
        //!
        //! @param params Vector of Parameter pointers to which this layer's parameters will be added.
        //!               The vector will have two entries added: weights and biases.
        //!
        //! @note The weights and biases must be initialized (via init_parameters) before calling this method.
        void collect_parameters(std::vector<Parameter*>& params) override;

        //! @brief Create a Dense layer using unique pointer.
        //!
        //! Factory method to create a Dense layer wrapped in std::unique_ptr.
        //!
        //! @param layerSize Number of neurons (output features) in the layer.
        //! @param func Name of the activation function.
        //! @return std::unique_ptr<LayerBase> Owned pointer to the created Dense layer.
        //!
        //! @see createUnique is a convenience factory for creating layers in ownership-transfer contexts.
        static std::unique_ptr<LayerBase> createUnique(int layerSize, const char* func);

    private:
        Parameter weights_;
        Parameter biases_;
    };
} // yann::models::layers
//! @}
