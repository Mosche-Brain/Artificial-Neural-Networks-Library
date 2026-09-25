 //
// Created by jaro on 7/24/26.
//

#include "yann/models/layers/Flatten.hpp"

#include "yann/runtime_config.hpp"

/*
 * Trzeba rozwiązać zachowanie wymiaru batcha przy spłaszczaniu próbek
 */

namespace yann::models::layers
{

    Flatten::Flatten(const cum::Shape& shape)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        YANN_LOG(1, "Initializing flatten layer with some input shape", "");
        #endif

        shape_ = shape;

        this->_layerType_ = LayerType::Flatten;
    }

    cum::Tensor Flatten::forward(const cum::Tensor& input)
    {
        return input.reshape(output_shape_);
    }

    cum::Tensor Flatten::backward(const cum::Tensor& deltaOutput)
    {
        return deltaOutput.reshape(input_shape_);
    }
} // yann::models::layers