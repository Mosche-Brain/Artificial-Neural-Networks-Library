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

        cum::Tensor forward(const cum::Tensor& input) override;
        cum::Tensor backward(const cum::Tensor& deltaOutput) override;

        void collect_parameters(std::vector<Parameter*>& params) override;

        cum::Tensor& weights() override { return weights_.values; }
        cum::Tensor& weights_grad() override { return weights_.gradient; }

        cum::Tensor& biases() override { return biases_.values; }
        cum::Tensor& biases_grad() override { return biases_.gradient; };

        static std::unique_ptr<LayerBase> createUnique(int layerSize);

    private:
        Parameter weights_;
        Parameter biases_;
    };
} // yann

#endif //YANN_LINEAR_HPP
