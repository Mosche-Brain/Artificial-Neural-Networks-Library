//
// Created by jaro on 8/31/26.
//

#ifndef YANN_CONV2D_HPP
#define YANN_CONV2D_HPP

#include <vector>

#include "cum/Core.hpp"
#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    class Conv2D : public LayerBase // NCHW layout
    {
    public:
        Conv2D(const cum::Shape& input_shape, cum::dim_t kernels_n, const cum::Shape& kernel_size, const char* activation, cum::dim_t stride=1, cum::dim_t padding=1, cum::dim_t dilatation=1);
        ~Conv2D();

        void initParameters(int output_features, int input_features) override {};

        cum::Tensor forward(const cum::Tensor& input) override;
        cum::Tensor backward(const cum::Tensor& input) override;

        cum::Matrix forward(const cum::Matrix& input) override {}
        cum::Matrix backward(const cum::Matrix& input) override {}

        void collect_parameters(std::vector<Parameter*>& params) override {};

        std::unique_ptr<LayerBase> createUnique(cum::Shape input_shape, cum::dim_t kernels_n, cum::Shape kernel_size, const char* activation, cum::dim_t stride=1, cum::dim_t padding=1, cum::dim_t dilatation=1);
    private:
        struct Config { cum::dim_t kernels_n; cum::Shape kernel_size; cum::dim_t stride; cum::dim_t padding; cum::dim_t dilatation; }
        config;

        std::vector<Parameter> kernels;
    };
} // yann

#endif //YANN_CONV2D_HPP
