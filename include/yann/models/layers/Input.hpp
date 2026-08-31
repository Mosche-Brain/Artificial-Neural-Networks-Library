                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    #pragma once
#include "LayerBase.hpp"

namespace yann::models::layers
{
    class Input : public LayerBase
    {
    public:
        Input(int layerSize);

        void initParameters(int output_features, int input_features) override;

        cum::Matrix forward(const cum::Matrix& input) override;
        cum::Matrix backward(const cum::Matrix& deltaOutput) override;

        void collect_parameters(std::vector<Parameter*>& params) override;

        static std::unique_ptr<LayerBase> createUnique(int layerSize);
    };
}