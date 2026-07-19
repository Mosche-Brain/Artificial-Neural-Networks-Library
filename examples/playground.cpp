#include <iostream>
#include <math.h>
#include <cum/cum.hpp>
#include <cum/Matrix.hpp>
#include <cum/LinearAlgebra.hpp>

#include <YANN//Models/Sequential.hpp>
#include <YANN/optimizers/SGD.hpp>
#include <YANN/runtime_config.hpp>

#include <matplot/matplot.h>

#include "helpers/conversion_helpers.hpp"

namespace plt = matplot;

int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);

    yann::runtime_config::set_verbosity(1);
    
    yann::models::Sequential model({
        yann::models::layers::Input::createUnique(1),
        yann::models::layers::Dense::createUnique(20, "tanh"),
        yann::models::layers::Dense::createUnique(20, "tanh"),
        // yann::models::layers::Dense::createUnique(48, "leaky_relu"),
        // yann::models::layers::Dense::createUnique(48, "leaky_relu"),
        // yann::models::layers::Dense::createUnique(128, "tanh"),
        yann::models::layers::Dense::createUnique(1, "linear"),
    });

    cum::cumeric_t x_min = -4.0 * M_PIf;
    cum::cumeric_t x_max =  4.0 * M_PIf;
    std::size_t N_train = 64;
    std::size_t N_eval = 512;

    cum::Matrix X_train = cum::Matrix::Linspace(x_min, x_max, N_train).transpose();
    cum::Matrix X_eval = cum::Matrix::Linspace(x_min, x_max, N_eval).transpose();
    cum::Matrix Y_train = cum::Matrix::Linspace(x_min, x_max, N_train).transpose();
    cum::Matrix Y_eval = cum::Matrix::Linspace(x_min, x_max, N_eval);

    cum::LinearAlgebra::sinInPlace(Y_train.data(), N_train);

    // normalize data

    // X_train /= x_max;
    // X_eval /= x_max;

    // model.setLossFunction(yann::utils::loss::LossFunction::binary_cross_entropy);
    // model.setLossFunction(yann::utils::loss::LossFunction::mse);
    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(0.1);
    // model.fit(X_train, Y_train, 0.01_c, 50);
    model.fit(X_train, Y_train, *optimizer, 50);

    cum::Matrix Y_pred = cum::Matrix(N_eval, 1);

    yann::runtime_config::set_verbosity(0);
    for (std::size_t i = 0 ; i < N_eval ; ++i)
    {
        cum::Matrix x(1, 1, {X_eval(i, 0)});
        Y_pred(i, 0) = model.forward(x)(0,0);
    }

    /* Plotting results */
    std::vector<double> X_train_plot = toStdVector<double>(X_train);
    std::vector<double> Y_train_plot = toStdVector<double>(Y_train);

    std::vector<double> X_eval_plot = toStdVector<double>(X_eval);
    std::vector<double> Y_pred_plot = toStdVector<double>(Y_pred);
    // std::vector<float> y_pred = toStdVector(model.predict(X_train));

    // plt::scatter(X_train_plot, Y_train_plot);
    plt::scatter(X_eval_plot, Y_pred_plot);


    // plt::plot(toStdVector(X_train), y_pred);
    plt::show();

    cum::decum();

    return 0;
}
