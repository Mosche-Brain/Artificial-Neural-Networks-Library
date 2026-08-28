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
        Flatten(int rows, int cols);

        cum::Matrix forward(const cum::Matrix& input) override;
        cum::Matrix backward(const cum::Matrix& deltaOutput) override;

    public:
        int input_rows;
        int input_cols;
        bool to_column_vector; // true by default
    };

} // yann::models::layers

#endif //YANN_FLATTEN_HPP
