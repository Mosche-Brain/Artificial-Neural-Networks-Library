#include <cum/cum.hpp>

#include <yann/loss/BinaryCrossEntropy.hpp>
#include <yann/models/Sequential.hpp>
#include <yann/models/layers/Dense.hpp>
#include <yann/models/layers/Input.hpp>
#include <yann/optimizers/SGD.hpp>
#include <yann/runtime_config.hpp>

#include <print>
#include <numeric>
#include <chrono>
#include <vector>

#include "serialize.hpp"


int main()
{
    cum::cum(cum::DEVICE::CPU); //

    yann::models::Sequential sequential({
        yann::models::layers::Input::createUnique(2),
        yann::models::layers::Dense::createUnique(2, "tanh"),
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

    yann::runtime_config::set_verbosity(0);

    cum::Tensor Y_pred = sequential.forward(X);

    for (int i = 0 ; i < X.rows() ; i++)
        std::println("[{}, {}] -> {}", X(i, 0), X(i, 1), Y_pred(i, 0));


    sequential.fit(X, Y, *loss, *optimizer, 500, 4);
    std::println("---------------------------------------");
    cum::Tensor Y_pred_2 = sequential.forward(X);

    for (int i = 0 ; i < X.rows() ; i++)
        std::println("[{}, {}] -> {}", X(i, 0), X(i, 1), Y_pred_2(i, 0));
    std::println("===============serialization=================");
    yann::utils::model_serialize_to_safetensors(sequential, "xor_solver.safetensors");

    yann::models::Sequential model({
        yann::models::layers::Input::createUnique(2),
        yann::models::layers::Dense::createUnique(2, "tanh"),
        yann::models::layers::Dense::createUnique(1, "sigmoid"),
    });

    std::println("===============deserialization=================");
    yann::utils::model_deserialize_from_safetensors(model, "xor_solver.safetensors");

    cum::Tensor Y_pred_3 = model.forward(X);

    for (int i = 0 ; i < X.rows() ; i++)
        std::println("[{}, {}] -> {}", X(i, 0), X(i, 1), Y_pred_3(i, 0));

    return 0;
}
