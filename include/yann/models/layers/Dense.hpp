#pragma once

#include "LayerBase.hpp"

namespace yann::models::layers
{
    class Dense : public LayerBase
    {
    public:
        Dense(int layerSize, const char* func);

        void initParameters(int output_features, int input_features) override;

        cum::Tensor forward(const cum::Tensor& input) override;
        cum::Tensor backward(const cum::Tensor& deltaOutput) override;

        cum::Matrix forward(const cum::Matrix& input) override;
        cum::Matrix backward(const cum::Matrix& deltaOutput) override;

        void collect_parameters(std::vector<Parameter*>& params) override;

        cum::Matrix& weights() override { return weights_.values; }
        cum::Matrix& weights_grad() override { return weights_.gradient; }

        cum::Matrix& biases() override { return biases_.values; }
        cum::Matrix& biases_grad() override { return biases_.gradient; };

        static std::unique_ptr<LayerBase> createUnique(int layerSize, const char* func);

    private:
        Parameter weights_;
        Parameter biases_;
    };
}