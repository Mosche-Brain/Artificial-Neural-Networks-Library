                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    #pragma once
#include "LayerBase.hpp"

namespace yann::models::layers
{
    class Input : public LayerBase
    {
    public:
        Input(int layerSize);

        void initParameters(int output_features, int input_features) override;

        cum::Tensor forward(const cum::Tensor& input) override;
        cum::Tensor backward(const cum::Tensor& deltaOutput) override;

        void collect_parameters(std::vector<Parameter*>& params) override;

        static std::unique_ptr<LayerBase> createUnique(int layerSize);
    };
}