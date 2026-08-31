#pragma once

namespace yann::models::layers
{    
    enum class LAYER_TYPE
    {
        INPUT,
        LINEAR,
        DENSE,
        CONV2,
        FLATTEN,
        POOL
    };
}