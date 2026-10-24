/**
 * @file Pool.hpp
 * @author jaro
 * @brief Base/placeholder pooling layer declaration for spatial downsampling.
 * @date 2026-08-31
 */

#ifndef YANN_POOL_HPP
#define YANN_POOL_HPP

#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    /**
     * @class Pool
     * @brief Spatial pooling layer (e.g. max pooling or average pooling).
     */
    class Pool : public LayerBase
    {
    };
} // yann::models::layers

#endif //YANN_POOL_HPP
