/*
 * @author: jaro
 * @name:   layers_forward
 * @file:   tests/layers_forward.cpp
 * @date:   13 September 2026 23:06:08
 */

#include <catch2/catch_test_macros.hpp>

#include <cum/runtime.hpp>
#include <cum/Matrix.hpp>
#include <cum/cum.hpp>

#include <yann/models/layers/Dense.hpp>

#include <print>


#include <yann/models/Sequential.hpp>
#include <yann/runtime_config.hpp>

TEST_CASE("Dense forward")
{
    cum::cum(cum::DEVICE::GPU);


    cum::Tensor X( { 8, 8 }, cum::default_type, cum::layout::IO);
    std::println("Created X tensor");

    X.fill(1);
    std::println("Filled X tensor with ones");

    auto layer = yann::models::layers::Dense::createUnique(8, "relu");
    std::println("Created Dense layer");

    layer->init_parameters(4, 8);
    std::println("Initialized Dense layer parameters");

    cum::Tensor Y = layer->forward(X);
    std::println("Forward pass completed");

    // For now, it would be fine if runtime error didn't occur
    REQUIRE(true);

    cum::decum();
}

TEST_CASE("Sequential forward")
{
    cum::cum(cum::DEVICE::GPU);

    std::println("Created CUM context");

    yann::models::Sequential sequential({
        yann::models::layers::Input::createUnique(2),
        yann::models::layers::Dense::createUnique(2, "relu"),
        yann::models::layers::Dense::createUnique(1, "sigmoid"),
    });

    std::println("Created Sequential model");

    cum::Tensor x(cum::Shape{2, 1}, cum::default_type, cum::layout::IO);
    x.fill(0.2137);

    std::println("Created input tensor");

    yann::runtime_config::set_verbosity(5);

    sequential.forward(x);

    REQUIRE(true);

}