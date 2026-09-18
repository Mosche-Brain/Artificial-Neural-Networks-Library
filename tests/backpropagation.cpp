/*
 * @author: jaro
 * @name:   backpropagation
 * @file:   tests/backpropagation.cpp
 * @date:   17 September 2026 18:51:22
 */

#include <catch2/catch_test_macros.hpp>
#include <print>

#include <cum/runtime.hpp>
#include <cum/Matrix.hpp>
#include <cum/cum.hpp>

#include <yann/models/layers/Dense.hpp>
#include <yann/models/layers/Linear.hpp>

#include <yann/models/Sequential.hpp>
#include <yann/runtime_config.hpp>

#include "loss/MeanSquaredError.hpp"
#include "optimizers/SGD.hpp"

TEST_CASE("Dense backward")
{

    cum::cum(cum::DEVICE::GPU);

    cum::Tensor dy(cum::Shape{3, 1}, cum::default_type, cum::layout::IO);
    std::println("Created dy tensor");

    dy.fill(0.2137);
    std::println("Filled dy tensor with pope values");

    auto layer = yann::models::layers::Dense::createUnique(3, "relu");
    std::println("Created Dense layer");

    layer->init_parameters(3, 1);

    cum::Tensor sample({1, 1}, cum::default_type, cum::layout::IO);
    layer->forward(sample);


    yann::runtime_config::set_verbosity(5);
    cum::Tensor dx = layer->backward(dy);
    std::println("Backpropagation completed");

    REQUIRE(true);
}

TEST_CASE("Linear backward")
{
    cum::cum(cum::DEVICE::GPU);

    cum::Tensor dy(cum::Shape{3, 1}, cum::default_type, cum::layout::IO);
    std::println("Created dy tensor");

    dy.fill(0.2137);
    std::println("Filled dy tensor with pope values");

    auto layer = yann::models::layers::Linear::createUnique(3);
    std::println("Created Dense layer");

    layer->init_parameters(3, 1);

    cum::Tensor sample({1, 1}, cum::default_type, cum::layout::IO);
    layer->forward(sample);


    yann::runtime_config::set_verbosity(5);
    cum::Tensor dx = layer->backward(dy);
    std::println("Backpropagation completed");

    REQUIRE(true);
}

TEST_CASE("Sequential backward")
{
    cum::cum(cum::DEVICE::GPU);

    yann::models::Sequential sequential({
        yann::models::layers::Input::createUnique(2),
        yann::models::layers::Dense::createUnique(3, "relu"),
        yann::models::layers::Dense::createUnique(1, "sigmoid"),
    });

    cum::Tensor X(2, 4, cum::default_type, cum::layout::IO);
    X.fill(0.2137);

    cum::Tensor Y(1, 4, cum::default_type, cum::layout::IO);
    Y.fill(0.69);

    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(0.01);
    yann::loss::Loss loss = yann::loss::MeanSquaredError::create();


    yann::runtime_config::set_verbosity(5);
    sequential.fit(X, Y, *loss, *optimizer, 1);

    REQUIRE(true);
}

