//
// Created by jaro on 7/2/26.
//
#include <cum/neural_primitives.hpp>
#include <cum/Tensor.hpp>
#include <cum/cum.hpp>
#include <cum/runtime.hpp>

#include <print>


using namespace cum;

int main()
{
    cum::cum(DEVICE::GPU);

    // cum::neural_primitives::Descriptor desc({3, 3, 3}, cum::neural_primitives::Descriptor::data_type::FP16);

    std::println("ko");

    runtime::sync();
    Tensor A({1, 3, 3, 3}, datatype::FP16, layout::NCHW);

    Tensor B({1, 3, 3, 3}, datatype::FP16, layout::NCHW);

    Tensor C = A + B;

    std::println("płint");

    std::println("A.sum() = {}", A.sum());
    std::println("B.sum() = {}", B.sum());
    std::println("C.sum() = {}", C.sum());

    std::println("ok");

    return 0;
}