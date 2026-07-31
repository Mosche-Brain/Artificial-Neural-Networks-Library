#include <iostream>
#include <math.h>
#include <cum/cum.hpp>
#include <cum/Matrix.hpp>
#include <cum/LinearAlgebra.hpp>

#include <yann/models/Sequential.hpp>
#include <yann/optimizers/SGD.hpp>
#include <yann/runtime_config.hpp>

#include <matplot/matplot.h>

#include "cum/runtime.hpp"
#include "helpers/conversion_helpers.hpp"
#include "logging/LossTracker.hpp"

namespace plt = matplot;

int main()
{
    cum::cum(cum::CUM_DEVICE::CPU);

    yann::runtime_config::set_verbosity(0);
    
    yann::models::Sequential model({
        yann::models::layers::Input::createUnique(1),
        yann::models::layers::Dense::createUnique(24, "tanh"),
        yann::models::layers::Dense::createUnique(24, "tanh"),
        yann::models::layers::Dense::createUnique(24, "tanh"),
        // yann::models::layers::Dense::createUnique(48, "leaky_relu"),
        // yann::models::layers::Dense::createUnique(48, "leaky_relu"),
        // yann::models::layers::Dense::createUnique(48, "leaky_relu"),
        // yann::models::layers::Dense::createUnique(128, "tanh"),
        yann::models::layers::Dense::createUnique(1, "tanh"),
    });

    cum::cumeric_t x_min = -4.0 * M_PIf;
    cum::cumeric_t x_max =  4.0 * M_PIf;
    std::size_t N_train = 32;
    std::size_t N_eval = 512;

    cum::Matrix X_train = cum::Matrix::Linspace(x_min, x_max, N_train).transpose();
    cum::Matrix X_eval = cum::Matrix::Linspace(x_min, x_max, N_eval).transpose();
    cum::Matrix Y_train = cum::Matrix::Linspace(x_min, x_max, N_train).transpose();
    cum::Matrix Y_eval = cum::Matrix::Linspace(x_min, x_max, N_eval);
    cum::Matrix Y_pred = cum::Matrix(N_eval, 1);

    cum::runtime::sync();
    cum::functions::trigonometric::sin_in_place(Y_train.data(), N_train);
    cum::runtime::sync();

    // normalize data

    X_train /= x_max;
    X_eval /= x_max;

    //
    // for (std::size_t i = 0 ; i < N_eval ; ++i)
    // {
    //     cum::Matrix x(1, 1, {X_eval(i, 0)});
    //     Y_pred(i, 0) = model.forward(x)(0,0);
    // }


    // model.setLossFunction(yann::utils::loss::LossFunction::binary_cross_entropy);
    // model.setLossFunction(yann::utils::loss::LossFunction::mse);
    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(0.01);
    // model.fit(X_train, Y_train, 0.01_c, 50);



    yann::logging::LossTracker loss_tracker = yann::logging::LossTracker();

    std::array<yann::logging::ITrainingCallback*, 1> callbacks = { &loss_tracker };

    model.fit(X_train, Y_train, *optimizer, 800, callbacks);

    // cum::Matrix sample(1, 2, 1._c);


    std::vector<double> loss;
    std::vector<double> epoch_range;

    for (int i = 0 ; i < loss_tracker.getLossHistory().size() ; i++)
    {
        loss.push_back(loss_tracker.getLossHistory()[i]);
        epoch_range.push_back(i);
    }



    // cum::Matrix Y_pred = cum::Matrix(N_eval, 1);

    yann::runtime_config::set_verbosity(0);
    for (std::size_t i = 0 ; i < N_eval ; ++i)
    {
        cum::Matrix x(1, 1, {X_eval(i, 0)});
        cum::runtime::sync();
        Y_pred(i, 0) = model.forward(x)(0,0);
        cum::runtime::sync();
    }
    cum::runtime::sync();

    /* Plotting results */
    std::vector<double> X_train_plot = toStdVector<double>(X_train);
    std::vector<double> Y_train_plot = toStdVector<double>(Y_train);

    std::vector<double> X_eval_plot = toStdVector<double>(X_eval);
    std::vector<double> Y_pred_plot = toStdVector<double>(Y_pred);
    // std::vector<float> y_pred = toStdVector(model.predict(X_train));

    plt::scatter(X_train_plot, Y_train_plot);
    // plt::scatter(X_eval_plot, Y_pred_plot);


    // plt::plot(X_eval_plot, Y_pred_plot);
    // plt::plot(epoch_range, loss);
    plt::plot(epoch_range, loss);
    // plt::plot(X_train_plot, Y_pred_plot);
    plt::show();

    cum::decum();

    return 0;
}
