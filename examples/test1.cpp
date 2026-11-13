
#include <cum/cum.hpp>
#include <cum/Tensor.hpp>

#include <iostream>

import yann;

using namespace cum;

int main()
{
    cum::cum(yann::DEVICE::CPU);

    dim_t neurons = 3;
    dim_t input_features = 2;

    yann::models::layers::Linear layer1(neurons);
    yann::models::layers::Dense layer2(neurons, "relu");

    layer1.init_parameters(neurons, input_features);
    layer2.init_parameters(neurons, input_features);

    Tensor x = Tensor::Random({3, 2});


    Tensor y = layer1.forward(x);



    return 0;
}