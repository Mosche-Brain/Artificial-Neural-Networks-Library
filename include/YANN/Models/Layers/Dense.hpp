#pragma once

#include "LayerBase.hpp"

namespace yann::models::layers
{
    class Dense : public LayerBase
    {
    public:
        // Dense(int layerSize, int inputWidth, const char* func);
        Dense(int layerSize, const char* func);

        // cum::Vector forward(const cum::Vector& input) override;
        cum::Matrix forward(const cum::Matrix& input) override;
        cum::Matrix backward(const cum::Matrix& deltaOutput) override;
        void update_weights(cum::cumeric_t rate) override;
        // std::vector<Parameter> collect_params() override;

        void collect_parameters(std::vector<Parameter*>& params) override;

        static std::unique_ptr<LayerBase> createUnique(int layerSize, const char* func);

    };
}