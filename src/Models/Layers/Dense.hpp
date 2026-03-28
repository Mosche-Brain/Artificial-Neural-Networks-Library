#pragma once

#include "build_config.hpp"
#include "LayerBase.hpp"

namespace YANN::Models::Layers
{
    class Dense : public LayerBase
    {
    public:
        // Dense(int layerSize, int inputWidth, const char* func);
        Dense(int layerSize, const char* func);

        matrix_t forward(const matrix_t& input) override;
        matrix_t backward(const matrix_t& deltaOutput) override;
        void update_weights(numeric_t rate);

        static std::unique_ptr<LayerBase> createUnique(int layerSize, const char* func);

    protected:
        matrix_t deltaWeights;
        vector_t deltaBiases; 
    };
}