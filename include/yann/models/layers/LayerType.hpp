/**
 * @file LayerType.hpp
 * @brief Enumeration of supported neural network layer types in YANN.
 */

#pragma once

namespace yann::models::layers
{    
    /**
     * @enum LayerType
     * @brief Discriminator enum identifying specific layer implementations.
     */
    enum class LayerType
    {
        Input,   /**< Input placeholder layer. */
        Linear,  /**< Fully connected linear layer (without activation). */
        Dense,   /**< Fully connected dense layer with activation. */
        Conv2D,  /**< 2D spatial convolution layer. */
        Conv3D,  /**< 3D spatial/volumetric convolution layer. */
        Flatten, /**< Tensor multidimensional flattening layer. */
        Pool,    /**< Spatial pooling layer (e.g. Max/Average pooling). */
        Softmax  /**< Softmax probability normalization layer. */
    };
}
