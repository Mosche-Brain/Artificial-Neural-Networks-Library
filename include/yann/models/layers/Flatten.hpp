//
// Created by jaro on 7/24/26.
//

#ifndef YANN_FLATTEN_HPP
#define YANN_FLATTEN_HPP

#include "yann/models/layers/LayerBase.hpp"

namespace yann::models::layers
{
    class Flatten : public LayerBase
    {
    public:
        // Flatten(int rows, int cols);
        Flatten(const cum::Shape& shape);

        cum::Tensor forward(const cum::Tensor& input) override;
        cum::Tensor backward(const cum::Tensor& deltaOutput) override;

    private:
        cum::Shape shape_;
        // int input_ros;
        // int input_cols;
        // bool to_column_vector; // true by default
    };

} // yann::models::layers

#endif //YANN_FLATTEN_HPP
