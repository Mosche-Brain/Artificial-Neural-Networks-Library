#include "Network/Layers.hpp"

INIT_PARAMS Layers::Flatten(int height, int width)
{

}

INIT_PARAMS Layers::Tanh(int size)
{
    INIT_PARAMS params;
    params.size = size;
    params.activation = tanhf;
    params.derivative = tanh_prim;

    return params;
}

INIT_PARAMS Layers::Linear(int size)
{
    INIT_PARAMS params;
    params.size = size;
    params.activation = pass;
    params.derivative = pass_prim;

    return params;
}

INIT_PARAMS Layers::Sigmoid(int size)
{
    INIT_PARAMS params;
    params.size = size;
    params.activation = sigmoid;
    params.derivative = sigmoid_prim;

    return params;
}

INIT_PARAMS Layers::RELunit(int size)
{
    INIT_PARAMS params;
    params.size = size;
    params.activation = RELu;
    params.derivative = RELu_prim;

    return params;
}

// INIT_PARAMS Layers::Normalize(int size)
// {

// }