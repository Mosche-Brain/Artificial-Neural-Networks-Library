#include "Models/Layers.hpp"

Layers::INIT_PARAMS Layers::lambda(std::function<double(double)> f, int size)
{
    Layers::INIT_PARAMS params;
    params.size = size;
    params.activation = f;
    params.derivative = [=](double x) { return derivative(f, x); };

    return params;
}

Layers::INIT_PARAMS Layers::Flatten(int height, int width)
{

}

Layers::INIT_PARAMS Layers::Tanh(int size)
{
    Layers::INIT_PARAMS params;
    params.size = size;
    params.activation = tanhf;
    params.derivative = tanh_prim;

    return params;
}

Layers::INIT_PARAMS Layers::Linear(int size)
{
    Layers::INIT_PARAMS params;
    params.size = size;
    params.activation = pass;
    params.derivative = pass_prim;

    return params;
}

Layers::INIT_PARAMS Layers::Sigmoid(int size)
{
    Layers::INIT_PARAMS params;
    params.size = size;
    params.activation = sigmoid;
    params.derivative = sigmoid_prim;

    return params;
}

Layers::INIT_PARAMS Layers::RELunit(int size)
{
    Layers::INIT_PARAMS params;
    params.size = size;
    params.activation = RELu;
    params.derivative = RELu_prim;

    return params;
}

Layers::INIT_PARAMS Layers::Normalize(int size)
{
    
}