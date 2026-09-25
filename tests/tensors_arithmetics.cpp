/*
 * @author: jaro
 * @name:   tensors_arithmetics
 * @file:   tests/tensors_arithmetics.cpp
 * @date:   13 September 2026 18:52:27
 */

#include <print>

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
            REQUIRE_THAT(tensor.data<cum::cumeric_t>()[index++], Catch::Matchers::WithinRel(value, static_cast<cum::cumeric_t>(1e-5)));
    }
}

TEST_CASE("Tensor binary operators perform arithmetic")
{
    cum::cum(cum::DEVICE::CPU);
    cum::Tensor a({2, 2}, cum::default_type, cum::layout::IO);
    cum::Tensor b({2, 2}, cum::default_type, cum::layout::IO);
    cum::Tensor identity({2, 2}, cum::default_type, cum::layout::IO);
    a.data<cum::cumeric_t>()[0] = 1; a.data<cum::cumeric_t>()[1] = 2; a.data<cum::cumeric_t>()[2] = 3; a.data<cum::cumeric_t>()[3] = 4;
    b.fill(2);
    identity.data<cum::cumeric_t>()[0] = 1; identity.data<cum::cumeric_t>()[1] = 0;
    identity.data<cum::cumeric_t>()[2] = 0; identity.data<cum::cumeric_t>()[3] = 1;

    require_values(a + b, {3, 4, 5, 6});
    require_values(a - b, {-1, 0, 1, 2});
    require_values(a * identity, {1, 2, 3, 4});
    require_values(a / b, {0.5, 1, 1.5, 2});
    cum::decum();
}

TEST_CASE("Tensor rowwise_sum reduces each matrix row")
{
    cum::cum(cum::DEVICE::CPU);
    cum::Tensor tensor(cum::Shape{2, 3}, cum::default_type, cum::layout::IO);
    cum::cumeric_t* values = tensor.data<cum::cumeric_t>();
    // values[0] = 1; values[1] = 2; values[2] = 3;
    // values[3] = 4; values[4] = 5; values[5] = 6;

    for (int i = 0; i < tensor.rows(); ++i)
    {
        for (int j = 0; j < tensor.cols(); ++j)
        {
            tensor.at<cum::cumeric_t>({i, j}) = i * (tensor.rows() + 1) + j + 1;
        }
    }

    REQUIRE(tensor.at({0, 0}) == 1);
    REQUIRE(tensor.at({0, 1}) == 2);
    REQUIRE(tensor.at({0, 2}) == 3);
    REQUIRE(tensor.at({1, 0}) == 4);
    REQUIRE(tensor.at({1, 1}) == 5);
    REQUIRE(tensor.at({1, 2}) == 6);

    const auto result = tensor.rowwise_sum();

    for (int i = 0; i < result.rows(); ++i)
    {
        for (int j = 0; j < result.cols(); ++j)
        {
            float value = result.at<cum::cumeric_t>({i, j});
            std::print("{}", value);
            if (j != result.cols() - 1)
                std::print(", ", value);
        }
        std::print("\n");
    }

    REQUIRE(result.shape() == (cum::Shape{2, 1}));
    REQUIRE(result.at<cum::cumeric_t>({0, 0}) == 6);
    REQUIRE(result.at<cum::cumeric_t>({1, 0}) == 15);
    cum::decum();
}

TEST_CASE("Tensor scalar operators work in both operand orders")
{
    cum::cum(cum::DEVICE::GPU);
    cum::Tensor tensor(cum::Shape{2}, cum::default_type, cum::layout::X);
    tensor.data<cum::cumeric_t>()[0] = 2;
    tensor.data<cum::cumeric_t>()[1] = 4;

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
    cum::Tensor tensor(cum::Shape{2}, cum::default_type, cum::layout::X);
    cum::Tensor other(cum::Shape{2}, cum::default_type, cum::layout::X);
    tensor.fill(4);
    other.fill(2);

    tensor += other;
    std::println("add");
    require_values(tensor, {6, 6});
    tensor -= other;
    std::println("sub");
    require_values(tensor, {4, 4});
    tensor.cwise_product_in_place(other);
    std::println("mul");
    require_values(tensor, {8, 8});
    tensor /= other;
    std::println("div");
    require_values(tensor, {4, 4});
    cum::decum();
}

