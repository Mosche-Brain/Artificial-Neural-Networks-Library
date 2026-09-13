/*
 * @author: jaro
 * @name:   tensors_eltwise
 * @file:   tests/tensors_eltwise.cpp
 * @date:   13 September 2026 18:55:00
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include <cum/Tensor.hpp>
#include <cum/cum.hpp>
#include <cum/neural_primitives/elementwise.hpp>

namespace
{
    void require_values(const cum::Tensor& tensor, std::initializer_list<cum::cumeric_t> expected)
    {
        REQUIRE(tensor.lenght() == static_cast<cum::dim_t>(expected.size()));
        std::size_t index = 0;
        for(const auto value : expected)
            REQUIRE_THAT(tensor.data()[index++], Catch::Matchers::WithinRel(value, static_cast<cum::cumeric_t>(1e-5)));
    }
}

TEST_CASE("Tensor fill and cwiseProduct operate elementwise")
{
    cum::cum(cum::DEVICE::CPU);
    cum::Tensor a({4}, cum::default_type, cum::layout::X);
    cum::Tensor b({4}, cum::default_type, cum::layout::X);
    a.fill(3);
    b.fill(2);

    require_values(a, {3, 3, 3, 3});
    require_values(a.multiply(b), {6, 6, 6, 6});
    require_values(a.cwiseProduct(b), {6, 6, 6, 6});
    cum::decum();
}

TEST_CASE("Tensor square and square root support value and in-place forms")
{
    cum::cum(cum::DEVICE::CPU);
    cum::Tensor tensor({3}, cum::default_type, cum::layout::X);
    tensor.data()[0] = 1;
    tensor.data()[1] = 4;
    tensor.data()[2] = 9;

    require_values(tensor.square(), {1, 16, 81});
    require_values(tensor.sqrt(), {1, 2, 3});
    tensor.square_in_place();
    require_values(tensor, {1, 16, 81});
    tensor.sqrt_in_place();
    require_values(tensor, {1, 4, 9});
    cum::decum();
}

TEST_CASE("Tensor neural elementwise wrappers produce transformed output")
{
    cum::cum(cum::DEVICE::CPU);
    cum::Tensor input({4}, cum::default_type, cum::layout::X);
    input.data()[0] = -2;
    input.data()[1] = -1;
    input.data()[2] = 0;
    input.data()[3] = 2;
    cum::Tensor output({4}, cum::default_type, cum::layout::X);

    cum::neural_primitives::relu(output, input);
    require_values(output, {0, 0, 0, 2});

    cum::neural_primitives::sigmoid(output, input);
    REQUIRE_THAT(output.at({0}), Catch::Matchers::WithinRel(static_cast<cum::cumeric_t>(1 / (1 + std::exp(2.0))), static_cast<cum::cumeric_t>(1e-5)));
    REQUIRE_THAT(output.at({3}), Catch::Matchers::WithinRel(static_cast<cum::cumeric_t>(1 / (1 + std::exp(-2.0))), static_cast<cum::cumeric_t>(1e-5)));
    cum::decum();
}
