/*
 * @author: jaro
 * @name:   Normalize
 * @file:   include/yann/models/layers/Normalize.hpp
 * @date:   26 September 2026 09:03:40
 */

#pragma once

#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    class Normalize : public LayerBase
    {
    public:
        Normalize(cum::dim_t lenght);
        Normalize(const cum::Shape& shape);

        cum::Tensor forward(const cum::Tensor& input) override;
        cum::Tensor backward(const cum::Tensor& ) override;

        std::unique_ptr<LayerBase> createUnique(const cum::Shape& shape);

    private:
    };
} // yann::models::layers

