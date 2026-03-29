#pragma once

#include "math_api.hpp"
#include "LayerBase.hpp"

namespace YANN::Models::Layers
{
    class Input : public LayerBase
    {
    public:
        Input(int layerSize);

        matrix_t forward(const matrix_t& input) override;
        matrix_t backward(const matrix_t& deltaOutput);
        void update_weights(numeric_t rate);

        static std::unique_ptr<LayerBase> createUnique(int layerSize);
    };
}