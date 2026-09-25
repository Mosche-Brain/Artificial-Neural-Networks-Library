//
// Created by jaro on 8/31/26.
//

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

TEST_CASE("MLP binary classifier for XOR")
{
    cum::cum(cum::DEVICE::CPU);

    yann::models::Sequential sequential({
        yann::models::layers::Input::createUnique(2),
        yann::models::layers::Dense::createUnique(3, "sigmoid"),
        yann::models::layers::Dense::createUnique(1, "sigmoid"),
    });

    cum::Tensor X(2, 4, cum::default_type, cum::layout::AB);
    X.at({0, 0}) = 0; X.at({1, 0}) = 0;;
    X.at({0, 1}) = 0; X.at({1, 1}) = 1;
    X.at({0, 2}) = 1; X.at({1, 2}) = 0;
    X.at({1, 1}) = 1; X.at({0, 3}) = 1;

    cum::Tensor Y(1, 4, cum::default_type, cum::layout::AB);
    Y.at({0, 0}) = 0; Y.at({1, 0}) = 1;
    Y.at({0, 1}) = 1; Y.at({1, 1}) = 0;

    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(0.01);
    yann::loss::Loss loss = yann::loss::MeanSquaredError::create();

    yann::runtime_config::set_verbosity(1);

    cum::Tensor Y_pred = sequential.forward(X);

    for (int i = 0 ; i < X.cols() ; i++)
        std::println("[{}, {}] -> {}", X(0, i), X(1, i), Y_pred(i, 0));


    sequential.fit(X, Y, *loss, *optimizer, 300);
    std::println("---------------------------------------");
    cum::Tensor Y_pred_2 = sequential.forward(X);

    for (int i = 0 ; i < X.cols() ; i++)
        std::println("[{}, {}] -> {}", X(0, i), X(1, i), Y_pred_2(i, 0));

    REQUIRE(true);
}

