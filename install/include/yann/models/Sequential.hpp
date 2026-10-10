//! @file Sequential.hpp
//! @brief Sequential model container that arranges layers in a linear topological structure.
//!
//! The Sequential class provides a simple way to create feed-forward neural networks
//! by arranging layers in a sequential topology. Layers are executed in the order they
//! are added, with the output of one layer serving as the input to the next.
//!
//! @note All layers added to a Sequential model must be properly initialized before use.
//!       The model handles forward and backward propagation through all layers automatically.
//!
//! @defgroup yann_sequential Sequential Model
//! @{

//! @brief Sequential model container for feed-forward neural networks.
//!
//! The Sequential class arranges layers in a linear topological structure, executing them
//! in the order they are added. It provides a high-level interface for training and
//! inference with multi-layer perceptrons.
//!
//! @tparam Layer Types derived from LayerBase can be added to the topology.
//!
//! @example
//! @code
//! yann::models::Sequential model;
//! model.addLayer(std::make_unique<yann::models::layers::Dense>(64, "relu"));
//! model.addLayer(std::make_unique<yann::models::layers::Dense>(10, "softmax"));
//! @endcode
//!
//! @see LayerBase
//! @see Dense
//! @see Input
//! @see Linear

#pragma once

#include "yann/logging/ITrainingCallback.hpp"
#include "yann/models/layers/LayerBase.hpp"
#include "yann/optimizers/OptimizerBase.hpp"
#include "yann/loss/LossBase.hpp"
#include "yann/Parameter.hpp"

namespace yann::models
{
    class Sequential
    {
    public:
        //! @brief Type alias for topology vector containing owned layer pointers.
        using Topology = std::vector<std::unique_ptr<layers::LayerBase>>;
        //! @brief Type alias for unique pointer to layer base.
        using LayerPtr = std::unique_ptr<layers::LayerBase>;

        //! @brief Default constructor. Creates an empty sequential model.
        Sequential() ;

        //! @brief Construct a Sequential model with initial topology.
        //!
        //! @param newTopology Initializer list of layers to add to the model.
        //! @param build If true (default), immediately builds the model structure.
        //! @note Layers are added in the order provided. The first layer receives input,
        //!       and the last layer produces the output.
        Sequential(std::initializer_list<LayerPtr> newTopology, bool build=true);

        //! @brief Perform forward pass through all layers.
        //!
        //! @param input Input tensor fed into the first layer.
        //! @return Output tensor produced by the last layer in the topology.
        //!
        //! @note The input flows through each layer sequentially. Intermediate outputs
        //!       are cached in each layer for use during backpropagation.
        cum::Tensor forward(const cum::Tensor& input);

        //! @brief Train the model on provided data.
        //!
        //! @param X Input tensor of shape (batch_size, input_features).
        //! @param Y Target tensor of shape (batch_size, output_features).
        //! @param loss Loss function to minimize during training.
        //! @param optimizer Optimizer to update model parameters.
        //! @param epochs Number of training iterations over the full dataset.
        //! @param batch_size Size of mini-batches (default: 1 for stochastic training).
        //! @param callbacks Optional training callbacks for logging/monitoring.
        //!
        //! @note Training proceeds through: forward pass -> loss computation -> backward pass
        //!       -> optimizer step -> parameter update. Gradients are accumulated through
        //!       all layers via backpropagation.
        void fit(const cum::Tensor& X, const cum::Tensor& Y, loss::LossBase& loss, optimizers::OptimizerBase& optimizer, cum::dim_t epochs, cum::dim_t batch_size=1, std::span<logging::ITrainingCallback*> callbacks = {});

        //! @brief Add a layer to the sequential topology.
        //!
        //! @param layer Unique pointer to a LayerBase-derived layer to append.
        //! @note The layer is taken ownership of and will be destroyed when the Sequential
        //!       model is destroyed. Layers are executed in addition order.
        void addLayer(LayerPtr layer);

        //! @brief Clear all layers from the topology.
        void clear();

        //! @brief Build the model topology (reinitialize shapes/parameters).
        void build();

        //! @brief Get weights of a specific layer.
        //!
        //! @param layer Index of the layer in the topology.
        //! @return Reference to the weight tensor of the specified layer.
        cum::Tensor& getWeights(size_t layer) const;

        //! @brief Get biases of a specific layer.
        //!
        //! @param layer Index of the layer in the topology.
        //! @return Reference to the bias tensor of the specified layer.
        cum::Tensor& getBiases(size_t layer) const;

        //! @brief Get outputs of a specific layer.
        //!
        //! @param layer Index of the layer in the topology.
        //! @return Reference to the output tensor of the specified layer.
        cum::Tensor& getOutputs(size_t layer) const;

        //! @brief Get activation function of a specific layer.
        //!
        //! @param layer Index of the layer in the topology.
        //! @return The activation function type of the specified layer.
        cum::functions::activation_t getActivation(size_t layer) const;

        //! @brief Get the complete topology (all layers).
        //!
        //! @return Const reference to the vector of layer pointers.
        auto getTopology() const -> const Topology&;

        //! @brief Get a specific layer by index.
        //!
        //! @param layer Index of the layer in the topology.
        //! @return Unique pointer to the layer at the specified position.
        auto getLayer(size_t layer) const -> LayerPtr;

        //! @brief Get the number of layers in the topology.
        //!
        //! @return Count of layers currently in the model.
        size_t getLayersCount() const;

        //! @brief Get the size (neuron count) of a specific layer.
        //!
        //! @param layer Index of the layer in the topology.
        //! @return Number of neurons in the specified layer.
        int getLayerSize(size_t layer) const;

        //! @brief Get all trainable parameters from the model.
        //!
        //! @return Vector of Parameter pointers containing weights and biases from all layers.
        std::vector<Parameter*> parameters();

    protected:
        //! @brief Backward pass through all layers.
        //!
        //! @param d_output Gradient from the loss function with respect to the output.
        //! @note Gradients are propagated in reverse order through all layers.
        void backward(const cum::Tensor& d_output);

        Topology topology;
    };
} // yann::models
//! @}
