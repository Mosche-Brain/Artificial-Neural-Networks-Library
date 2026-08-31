//
// Created by jaro on 8/31/26.
//

#ifndef YANN_LINEAR_HPP
#define YANN_LINEAR_HPP

#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    class Linear : public LayerBase
    {
    public:
        Linear(int layerSize);

        void initParameters(int output_features, int input_features) override;

        cum::Matrix forward(const cum::Matrix& input) override;
        cum::Matrix backward(const cum::Matrix& deltaOutput) override;

        void collect_parameters(std::vector<Parameter*>& params) override;

        cum::Matrix& weights() override { return weights_.values; }
        cum::Matrix& weights_grad() override { return weights_.gradient; }

        cum::Matrix& biases() override { return biases_.values; }
        cum::Matrix& biases_grad() override { return biases_.gradient; };

        static std::unique_ptr<LayerBase> createUnique(int layerSize);

    private:
        Parameter weights_;
        Parameter biases_;
    };
} // yann

#endif //YANN_LINEAR_HPP
