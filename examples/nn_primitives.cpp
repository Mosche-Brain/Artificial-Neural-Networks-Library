//
// Created by jaro on 7/2/26.
//
#include <cum/neural_primitives.hpp>
#include <cum/cum.hpp>

int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);

    cum::neural_primitives::tensor_descriptor desc({3, 3, 3}, cum::neural_primitives::tensor_descriptor::data_type::FP16);


    return 0;
}