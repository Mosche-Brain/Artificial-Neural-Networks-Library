/**
 * @file LayerBase.hpp
 * @brief Base class for neural network layers and layer caching structures in YANN.
 */

#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include <cum/Matrix.hpp>
#include <cum/Vector.hpp>
#include <cum/LinearAlgebra.hpp>
#include <cum/functions.hpp>
#include <cum/Tensor.hpp>

#include "yann/models/layers/Kind.hpp"
#include "yann/Parameter.hpp"
#include "yann/common.hpp"

/*
 * Activation should be moved to declaration of specyfic layer due to it is not universal feature for all layer types
 * ForwardCache name may not be relevant due to storing dz and da, more general name like a `LayerCache` may be better
 * TODO: Do not implicitly reserve memory necessary for training (da, dz, etc)
 */

namespace cum::neural_primitives
{
    enum class Purpose;
}

namespace yann::models::layers
{
    /**
     * @struct LayerCache
     * @brief Storage cache for intermediate activations and gradients during forward and backward passes.
     */
    struct LayerCache
    {
        std::unordered_set<cum::Tensor*> owned;
        cum::Tensor x;  /**< Input tensor to the layer. */

        LayerCache() = default;

        /**
         * @brief Fill all cached tensors with zeros.
         */
        void clear()
        {
            for (auto& tensor : owned)
            {
                tensor->fill(0);
            }
        }

        /**
         * @brief cash
         */
        void cast(cum::datatype dtype)
        {
            for (auto& tensor : owned)
                tensor->cast(dtype);
        }

        /**
         * @brief Resize 2D matrices in the cache for 2D fully connected layers.
         * @param in_features Number of input features.
         * @param out_features Number of output features.
         * @param batch_size Current batch size dimension.
         */
        void resize(cum::dim_t in_features, cum::dim_t out_features, cum::dim_t batch_size)
        {
            // x   = cum::Tensor::make_matrix(batch_size, in_features);
            // z   = cum::Tensor::make_matrix(batch_size, out_features);
            // a   = cum::Tensor::make_matrix(batch_size, out_features);
            // dz  = cum::Tensor::make_matrix(batch_size, out_features);
        }

        /**
         * @brief Resize cache tensors for arbitrary N-D tensor shapes.
         * @param input_shape Shape of inputs (excluding batch dimension).
         * @param output_shape Shape of outputs (excluding batch dimension).
         * @param batch_size Current batch size prepended to shapes.
         */
        void resise(const cum::Shape& input_shape, const cum::Shape& output_shape, cum::dim_t batch_size)
        {
            // cum::Shape in_shape = input_shape;
            // cum::Shape out_shape = output_shape;
            //
            // // batch is always the first (0) dimension
            // in_shape.insert(in_shape.begin(), batch_size);
            // out_shape.insert(out_shape.begin(), batch_size);
            //
            // x   = cum::Tensor(in_shape);
            // z   = cum::Tensor(out_shape);
            // a   = cum::Tensor(out_shape);
            // dz  = cum::Tensor(out_shape);
            // da  = cum::Tensor(out_shape);
        }
    };

    /**
     * @class LayerBase
     * @brief Abstract base class defining the common interface for all neural network layers.
     */
    class LayerBase
    {
    public:
        enum class Mode { inference, training };

        virtual ~LayerBase() = default;

        /**
         * @brief Initialize trainable layer parameters given 2D feature dimensions.
         * @param output_features Number of output features / neurons.
         * @param input_features Number of input features.
         */
        virtual void init_parameters(int output_features, int input_features);

        /**
         * @brief Initialize trainable layer parameters given multi-dimensional shapes.
         * @param input_shape Input tensor shape specification.
         * @param output_shape Output tensor shape specification.
         */
        virtual void init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape);

        /**
         * @brief Get the configured input shape for this layer.
         * @return Const reference to input cum::Shape.
         */
        [[nodiscard]] virtual const cum::Shape& input_shape() const;

        /**
         * @brief Get the configured output shape produced by this layer.
         * @return Const reference to output cum::Shape.
         */
        [[nodiscard]] virtual const cum::Shape& output_shape() const;

        /**
         * @brief Compute the forward propagation through the layer.
         * @param input Input tensor to the layer.
         * @return Output tensor produced by the layer.
         */
        virtual cum::Tensor forward(const cum::Tensor& input) = 0;

        /**
         * @brief Compute the backward propagation through the layer.
         * @param gradient Gradient flowing back from the subsequent layer (dL/dy).
         * @return Gradient with respect to the layer input (dL/dx).
         */
        virtual cum::Tensor backward(const cum::Tensor& gradient) = 0;

        /**
         * @brief Collect pointers to all trainable parameters owned by this layer.
         * @param params Vector to append Parameter pointers into.
         */
        virtual void collect_parameters(std::vector<Parameter*>& params) = 0;

        /**
         * @brief Check whether layer parameters and shapes have been initialized.
         * @return True if initialized, false otherwise.
         */
        bool initialized();

        /**
         * @brief Get the size / number of units in the layer.
         * @return Number of output units.
         */
        virtual int size();

        /**
         * @brief Get reference to the cached output activation tensor.
         * @return Reference to output activation tensor.
         */
        virtual cum::Tensor& Outputs();

        /**
         * @brief Get reference to the cached input tensor.
         * @return Reference to input tensor.
         */
        virtual cum::Tensor& Inputs();

        /**
         * @brief Get reference to the weight gradients tensor.
         * @return Reference to weight gradients tensor.
         */
        virtual cum::Tensor& weights_grad();

        /**
         * @brief Get reference to the weights tensor.
         * @return Reference to weights tensor.
         */
        virtual cum::Tensor& weights();

        /**
         * @brief Get reference to the biases tensor.
         * @return Reference to biases tensor.
         */
        virtual cum::Tensor& biases();

        /**
         * @brief Get reference to the bias gradients tensor.
         * @return Reference to bias gradients tensor.
         */
        virtual cum::Tensor& biases_grad();

        cum::functions::activation_t activation; /**< Activation function applied by the layer. */
        
        /**
         * @brief Get the concrete layer type discriminator.
         * @return LayerType enum value identifying this layer.
         */
        Kind layerType();

        using Cache = LayerCache;
    protected:
        friend class Sequential;

        Mode _mode = Mode::training;

        cum::Shape input_shape_;  /**< Shape of inputs accepted by the layer. */
        cum::Shape output_shape_; /**< Shape of outputs produced by the layer. */

        bool _initialized = false; /**< Initialization status flag. */
        Kind _kind;      /**< Layer type enumeration value. */
    };
} // yann::models::layers
