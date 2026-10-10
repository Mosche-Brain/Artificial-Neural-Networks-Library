/*
 * @author: jaro
 * @name:   Softmax
 * @file:   include/yann/models/layers/Softmax.hpp
 * @date:   02 October 2026 17:18:23
 */

#pragma once

#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    class Softmax : public LayerBase
    {
    public:
        Softmax(cum::dim_t lenght, cum::dim_t axis = -1);
        Softmax(const cum::Shape& shape, cum::dim_t axis = -1);

        void init_parameters(int output_features, int input_features) override;
        void init_parameters(const cum::Shape& input_shape, const cum::Shape& output_shape) override;

        cum::Tensor forward(const cum::Tensor& input) override;
        cum::Tensor backward(const cum::Tensor& gradient) override;

        void collect_parameters(std::vector<Parameter*>& params) override;

        static std::unique_ptr<LayerBase> createUnique(cum::dim_t lenght, cum::dim_t axis = -1);
        static std::unique_ptr<LayerBase> createUnique(const cum::Shape& shape, cum::dim_t axis = -1);
    private:
        LayerCache cache;

        cum::dim_t axis_;
    };
} // yann

