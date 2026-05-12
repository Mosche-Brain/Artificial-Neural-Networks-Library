#pragma once

#include "LayerBase.hpp"

namespace YANN::Models::Layers
{
    class Dense : public LayerBase
    {
    public:
        // Dense(int layerSize, int inputWidth, const char* func);
        Dense(int layerSize, const char* func);

        cum::Matrix forward(const cum::Matrix& input) override;
        cum::Matrix backward(const cum::Matrix& deltaOutput) override;
        void update_weights(cum::cumeric_t rate);

        static std::unique_ptr<LayerBase> createUnique(int layerSize, const char* func);

    protected:
        cum::Matrix deltaWeights;
        vector_t deltaBiases; 
    };
}