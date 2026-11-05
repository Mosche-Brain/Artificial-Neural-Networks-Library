//
// Created by jaro on 8/31/26.
//

#include <catch2/catch_test_macros.hpp>
#include <print>

#include <cum/Matrix.hpp>
#include <cum/cum.hpp>

#include <yann/models/layers/Dense.hpp>
#include <yann/models/layers/Input.hpp>

#include <yann/models/Chain.hpp>
#include <yann/models/Sequential.hpp>
#include <yann/runtime_config.hpp>

#include <yann/loss/BinaryCrossEntropy.hpp>
#include <yann/optimizers/SGD.hpp>


TEST_CASE("MLP binary classifier for XOR")
{
    cum::cum(cum::DEVICE::CPU); //

    yann::models::Sequential sequential({
        yann::models::layers::Input::createUnique(2),
        yann::models::layers::Dense::createUnique(3, "tanh"),
        yann::models::layers::Dense::createUnique(1, "sigmoid"),
    });

    cum::Tensor X(4, 2, cum::default_type, cum::layout::AB);
    X.at({0, 0}) = 0; X.at({0, 1}) = 0;;
    X.at({1, 0}) = 0; X.at({1, 1}) = 1;
    X.at({2, 0}) = 1; X.at({2, 1}) = 0;
    X.at({3, 0}) = 1; X.at({3, 1}) = 1;

    cum::Tensor Y(4, 1, cum::default_type, cum::layout::AB);
    Y.at({0, 0}) = 0;
    Y.at({1, 0}) = 1;
    Y.at({2, 0}) = 1;
    Y.at({3, 0}) = 0;

    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(0.4); // BatchMean loss scaling: lr * batch_size (4) keeps the previous step size
    yann::loss::Loss loss = yann::loss::BinaryCrossEntropy::create();

    yann::runtime_config::set_verbosity(1);

    cum::Tensor Y_pred = sequential.forward(X);

    // for (int i = 0 ; i < X.rows() ; i++)
        // std::println("[{}, {}] -> {}", X(i, 0), X(i, 1), Y_pred(i, 0));


    sequential.fit(X, Y, *loss, *optimizer, 500, 4);
    // std::println("---------------------------------------");
    cum::Tensor Y_pred_2 = sequential.forward(X);

    // for (int i = 0 ; i < X.rows() ; i++)
        // std::println("[{}, {}] -> {}", X(i, 0), X(i, 1), Y_pred_2(i, 0));

    REQUIRE(static_cast<int>(std::round(Y_pred_2(0, 0))) == 0);
    REQUIRE(static_cast<int>(std::round(Y_pred_2(1, 0))) == 1);
    REQUIRE(static_cast<int>(std::round(Y_pred_2(2, 0))) == 1);
    REQUIRE(static_cast<int>(std::round(Y_pred_2(3, 0))) == 0);
}

TEST_CASE("MLP reggresion")
{
    // todo
    REQUIRE(true);
}

TEST_CASE("Chain jointly trains two Sequential models (encoder -> decoder)")
{
    cum::cum(cum::DEVICE::CPU);

    yann::models::Sequential encoder({
        yann::models::layers::Input::createUnique(2),
        yann::models::layers::Dense::createUnique(3, "tanh"),
    });

    yann::models::Sequential decoder({
        yann::models::layers::Input::createUnique(3),
        yann::models::layers::Dense::createUnique(1, "sigmoid"),
    });

    yann::models::Chain chain(encoder, decoder);
    REQUIRE(chain.size() == 2);
    REQUIRE(chain.parameters().size() == 4);

    cum::Tensor X(4, 2, cum::default_type, cum::layout::AB);
    X.at({0, 0}) = 0; X.at({0, 1}) = 0;
    X.at({1, 0}) = 0; X.at({1, 1}) = 1;
    X.at({2, 0}) = 1; X.at({2, 1}) = 0;
    X.at({3, 0}) = 1; X.at({3, 1}) = 1;

    cum::Tensor Y(4, 1, cum::default_type, cum::layout::AB);
    Y.at({0, 0}) = 0;
    Y.at({1, 0}) = 1;
    Y.at({2, 0}) = 1;
    Y.at({3, 0}) = 0;

    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(0.4);
    yann::loss::Loss loss = yann::loss::BinaryCrossEntropy::create();

    yann::runtime_config::set_verbosity(1);

    chain.fit(X, Y, *loss, *optimizer, 500, 4);

    // Verify both chained forward and standalone encoder -> decoder forward match
    cum::Tensor Y_chain = chain.forward(X);
    cum::Tensor latent = encoder.forward(X);
    cum::Tensor Y_separate = decoder.forward(latent);

    REQUIRE(latent.shape() == (cum::Shape{4, 3}));
    REQUIRE(Y_chain == Y_separate);

    REQUIRE(static_cast<int>(std::round(Y_chain(0, 0))) == 0);
    REQUIRE(static_cast<int>(std::round(Y_chain(1, 0))) == 1);
    REQUIRE(static_cast<int>(std::round(Y_chain(2, 0))) == 1);
    REQUIRE(static_cast<int>(std::round(Y_chain(3, 0))) == 0);

    cum::decum();
}
