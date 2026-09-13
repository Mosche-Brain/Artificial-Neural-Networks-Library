/*
 * @author: jaro
 * @name:   tensors_arithmetics
 * @file:   tests/tensors_arithmetics.cpp
 * @date:   13 September 2026 18:52:27
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cum/Tensor.hpp>
#include <cum/cum.hpp>

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

TEST_CASE("Tensor binary operators perform arithmetic")
{
    cum::cum(cum::DEVICE::CPU);
    cum::Tensor a({2, 2}, cum::default_type, cum::layout::IO);
    cum::Tensor b({2, 2}, cum::default_type, cum::layout::IO);
    cum::Tensor identity({2, 2}, cum::default_type, cum::layout::IO);
    a.data()[0] = 1; a.data()[1] = 2; a.data()[2] = 3; a.data()[3] = 4;
    b.fill(2);
    identity.data()[0] = 1; identity.data()[1] = 0;
    identity.data()[2] = 0; identity.data()[3] = 1;

    require_values(a + b, {3, 4, 5, 6});
    require_values(a - b, {-1, 0, 1, 2});
    require_values(a * identity, {1, 2, 3, 4});
    require_values(a / b, {0.5, 1, 1.5, 2});
    cum::decum();
}

TEST_CASE("Tensor scalar operators work in both operand orders")
{
    cum::cum(cum::DEVICE::CPU);
    cum::Tensor tensor({2}, cum::default_type, cum::layout::X);
    tensor.data()[0] = 2;
    tensor.data()[1] = 4;

    require_values(tensor + 1, {3, 5});
    require_values(1 + tensor, {3, 5});
    require_values(tensor - 1, {1, 3});
    require_values(1 - tensor, {-1, -3});
    require_values(tensor * 2, {4, 8});
    require_values(2 * tensor, {4, 8});
    require_values(tensor / 2, {1, 2});
    require_values(8 / tensor, {4, 2});
    cum::decum();
}

TEST_CASE("Tensor compound operators mutate the left operand")
{
    cum::cum(cum::DEVICE::CPU);
    cum::Tensor tensor({2}, cum::default_type, cum::layout::X);
    cum::Tensor other({2}, cum::default_type, cum::layout::X);
    tensor.fill(4);
    other.fill(2);

    tensor += other;
    require_values(tensor, {6, 6});
    tensor -= other;
    require_values(tensor, {4, 4});
    tensor *= other;
    require_values(tensor, {8, 8});
    tensor /= other;
    require_values(tensor, {4, 4});
    cum::decum();
}

