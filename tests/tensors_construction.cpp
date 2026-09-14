/*
 * @author: jaro
 * @name:   tensor_construction
 * @file:   tests/tensor_construction.cpp
 * @date:   13 September 2026 18:54:34
 */


#include <catch2/catch_test_macros.hpp>


#include <cum/memory.hpp>
#include <utility>

#include <cum/Tensor.hpp>
#include <cum/cum.hpp>

TEST_CASE("Tensor constructor exposes shape, type and layout")
{
    cum::cum(cum::DEVICE::CPU);
    cum::Tensor tensor({2, 3}, cum::default_type, cum::layout::IO);
    tensor.fill(7);

    REQUIRE(tensor.shape() == (cum::Shape{2, 3}));
    REQUIRE(tensor.dims() == 2);
    REQUIRE(tensor.lenght() == 6);
    REQUIRE(tensor.size() == 6 * static_cast<cum::dim_t>(cum::datatype_size(cum::default_type)));
    REQUIRE(tensor.format() == cum::layout::IO);
    REQUIRE(tensor.type() == cum::default_type);
    REQUIRE(tensor.rows() == 2);
    REQUIRE(tensor.cols() == 3);
    REQUIRE(tensor.has(cum::Axis::Rows));
    REQUIRE(tensor.extent(cum::Axis::Rows) == 2);
    REQUIRE(tensor.extent(cum::Axis::Cols) == 3);
    REQUIRE(tensor.at({1, 2}) == 7);
    const cum::Tensor& const_tensor = tensor;
    REQUIRE(const_tensor.at({1, 2}) == 7);
    cum::decum();
}

TEST_CASE("Tensor copy, move and assignment preserve independent values")
{
    cum::cum(cum::DEVICE::CPU);
    cum::Tensor original({2, 2}, cum::default_type, cum::layout::IO);
    original.fill(3);

    cum::Tensor copy(original);
    copy.data()[0] = 8;
    REQUIRE(original.at({0, 0}) == 3);
    REQUIRE(copy.at({0, 0}) == 8);

    cum::Tensor assigned;
    assigned = original;
    REQUIRE(assigned.at({1, 1}) == 3);

    cum::Tensor moved(std::move(copy));
    REQUIRE(moved.at({0, 0}) == 3);
    cum::decum();
}

TEST_CASE("Tensor factories create requested shapes and initialization")
{
    cum::cum(cum::DEVICE::CPU);

    auto vector = cum::Tensor::make_vector(4);
    auto matrix = cum::Tensor::make_matrix(2, 3);
    auto scalar = cum::Tensor::make_scalar();
    auto typed = cum::Tensor::create_tensor<cum::default_type, cum::layout::X>({4});
    auto zeros = cum::Tensor::Zeros({2, 2}, cum::default_type, cum::layout::IO);
    auto ones = cum::Tensor::Ones({2, 2}, cum::default_type, cum::layout::IO);
    auto line = cum::Tensor::Linspace(1, 4, 4);

    REQUIRE(vector.shape() == (cum::Shape{4}));
    REQUIRE(matrix.shape() == (cum::Shape{2, 3}));
    REQUIRE(scalar.shape() == (cum::Shape{1}));
    REQUIRE(typed.shape() == (cum::Shape{4}));
    REQUIRE(zeros.sum() == 0);
    REQUIRE(ones.sum() == 4);
    REQUIRE(line.at({0}) == 1);
    REQUIRE(line.at({3}) == 4);
    cum::decum();
}

TEST_CASE("Tensor take_memory uses supplied storage")
{
    cum::cum(cum::DEVICE::CPU);
    auto* storage = cum::memory::allocate(3);
    storage[0] = 1;
    storage[1] = 2;
    storage[2] = 3;
    {
        auto tensor = cum::Tensor::take_memory({3}, storage, cum::default_type, cum::layout::X);
        tensor.data()[1] = 9;
        REQUIRE(storage[1] == 9);
        REQUIRE(tensor.at({2}) == 3);
    }
    cum::memory::free(storage);
    cum::decum();
}
