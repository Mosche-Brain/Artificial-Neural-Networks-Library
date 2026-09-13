//
// Created by jaro on 7/2/26.
//
#include <cum/neural_primitives.hpp>
#include <cum/neural_primitives/elementwise.hpp>
#include <cum/Tensor.hpp>
#include <cum/cum.hpp>
#include <cum/runtime.hpp>

#include <print>


using namespace cum;

int main()
{
    cum::cum(DEVICE::GPU);

    // cum::neural_primitives::Descriptor desc({3, 3, 3}, cum::neural_primitives::Descriptor::data_type::FP16);

    runtime::sync();
    Tensor W({1, 3, 3, 3}, datatype::BF16, layout::NCHW);

    Tensor X({1, 3, 3, 3}, datatype::BF16, layout::NCHW);

    W.fill(6);
    X.fill(9);

    Tensor Z = W * X;

    Tensor Y = Z;

    neural_primitives::tanh(Y);


    std::println("W.sum() = {}", W.sum());
    std::println("X.sum() = {}", X.sum());
    std::println("Z.sum() = {}", Z.sum());
    std::println("Y.sum() = {}", Y.sum());

    std::println("ok");

    return 0;
}