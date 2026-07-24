 //
// Created by jaro on 7/24/26.
//

#include "Flatten.hpp"

#include "runtime_config.hpp"

namespace yann::models::layers
{

    Flatten::Flatten(int rows, int cols)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        yann::logger().log(1, "Initializing flatten layer with {} x  input shape", rows);
        #endif
        input_rows = rows;
        input_cols = cols;
        to_column_vector = true;

        this->_layerType_ = LAYER_TYPE::FLATTEN;
    }

    cum::Matrix Flatten::forward(const cum::Matrix& input)
    {
        return input.reshape(input_rows * input_cols, 1);
    }

    cum::Matrix Flatten::backward(const cum::Matrix& deltaOutput)
    {
        return deltaOutput.reshape(input_rows, input_cols);
    }
} // yann::models::layers