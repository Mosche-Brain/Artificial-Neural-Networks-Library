//
// Created by jaro on 8/31/26.
//

#include "yann/models/layers/Conv2D.hpp"

namespace yann::models::layers
{

    Conv2D::Conv2D(const cum::Shape& input_shape, const cum::dim_t kernels_n, const cum::Shape& kernel_size, const char* activation, const cum::dim_t stride, const cum::dim_t padding, const cum::dim_t dilatation)
    {
        config = {
            .kernels_n=kernels_n,
            .kernel_size=kernel_size,
            .stride=stride,
            .padding=padding,
            .dilatation=dilatation
        };
    }

    Conv2D::~Conv2D()
    {

    }

    std::unique_ptr<LayerBase> Conv2D::createUnique(const cum::Shape input_shape, const cum::dim_t kernels_n, const cum::Shape kernel_size, const char* activation, const cum::dim_t stride, const cum::dim_t padding, const cum::dim_t dilatation)
    {
        return std::make_unique<Conv2D>(input_shape, kernels_n, kernel_size, activation, stride, padding, dilatation);
    }
  
} // yann