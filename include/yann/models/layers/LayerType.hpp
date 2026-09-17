#pragma once

namespace yann::models::layers
{    
    enum class LayerType
    {
        Input,
        Linear,
        Dense,
        Conv2D,
        Conv3D,
        Flatten,
        Pool
    };
}