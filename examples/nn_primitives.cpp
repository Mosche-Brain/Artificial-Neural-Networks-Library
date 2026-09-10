//
// Created by jaro on 7/2/26.
//
#include <cum/neural_primitives.hpp>
#include <cum/Tensor.hpp>
#include <cum/cum.hpp>

using namespace cum;

int main()
{
    cum::cum(DEVICE::GPU);

    // cum::neural_primitives::Descriptor desc({3, 3, 3}, cum::neural_primitives::Descriptor::data_type::FP16);

    Tensor tensor({3, 3, 3}, datatype::FP16);



    return 0;
}