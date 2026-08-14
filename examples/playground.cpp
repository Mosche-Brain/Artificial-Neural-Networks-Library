#include <cum/cum.hpp>
#include <cum/Matrix.hpp>
#include <cum/runtime.hpp>

#include <yann/models/Sequential.hpp>
#include <yann/optimizers/SGD.hpp>
#include <yann/runtime_config.hpp>
#include <yann/loss/MeanSquaredError.hpp>
#include <yann/logging/LossTracker.hpp>


#include <matplot/matplot.h>
#include "helpers/conversion_helpers.hpp"

#include <iostream>
#include <math.h>

#include "loss/BinaryCrossEntropy.hpp"

namespace plt = matplot;

int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);

    yann::runtime_config::set_verbosity(0);
    
    yann::models::Sequential model({
        yann::models::layers::Input::createUnique(1),
        yann::models::layers::Dense::createUnique(64, "tanh"),
        // yann::models::layers::Dense::createUnique(16, "tanh"),
        // yann::models::layers::Dense::createUnique(32, "relu"),
        yann::models::layers::Dense::createUnique(1, "linear"),
    });

    cum::cumeric_t x_min = -6.0 * M_PIf;
    cum::cumeric_t x_max =  6.0 * M_PIf;
    std::size_t N_train = 48;
    std::size_t N_eval = 512;

    cum::Matrix X_train = cum::Matrix::Linspace(x_min, x_max, N_train).transpose(); // linspace in row vector
    cum::Matrix Y_train = cum::Matrix::Linspace(x_min, x_max, N_train).transpose();


    cum::Matrix X_eval = cum::Matrix::Linspace(x_min, x_max, N_eval);
    cum::Matrix Y_eval = cum::Matrix::Linspace(x_min, x_max, N_eval);

    cum::runtime::sync();
    cum::functions::trigonometric::sin_in_place(Y_eval.data(), N_eval);
    cum::functions::trigonometric::sin_in_place(Y_train.data(), N_train);
    cum::runtime::sync();

    Y_train *= 0.5_c;
    Y_train += 0.5_c;

    Y_eval *= 0.5_c;
    Y_eval += 0.5_c;

    // X_train /= x_max;
    // X_eval /= x_max;

    cum::Matrix Y_pred_pretrained = model.forward(X_eval); // batch

    yann::loss::Loss loss = yann::loss::MeanSquaredError::create();
    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(0.01);


    yann::logging::LossTracker loss_tracker = yann::logging::LossTracker();

    std::array<yann::logging::ITrainingCallback*, 1> callbacks = { &loss_tracker };

    yann::runtime_config::set_verbosity(1);
    model.fit(X_train, Y_train, *loss, *optimizer, 12, callbacks);
    cum::runtime::sync();


    std::vector<double> loss_history;
    std::vector<double> epoch_range;

    for (int i = 0 ; i < loss_tracker.getLossHistory().size() ; i++)
    {
        loss_history.push_back(loss_tracker.getLossHistory()[i]);
        epoch_range.push_back(i);
    }


    yann::runtime_config::set_verbosity(0);


    cum::runtime::sync();

    cum::Matrix Y_pred = model.forward(X_eval); // batch

    cum::runtime::sync();

    /* Plotting results */
    std::vector<double> X_train_plot = toStdVector<double>(X_train);
    std::vector<double> Y_train_plot = toStdVector<double>(Y_train);

    std::vector<double> X_eval_plot = toStdVector<double>(X_eval);
    std::vector<double> Y_eval_plot = toStdVector<double>(Y_eval);
    std::vector<double> Y_pred_plot = toStdVector<double>(Y_pred);
    std::vector<double> Y_pred_pretrained_plot = toStdVector<double>(Y_pred_pretrained);

    plt::gcf()->size(1400, 900);

    plt::tiledlayout(2, 2);
    auto ax1 = plt::nexttile();
    plt::plot(ax1, X_eval_plot, Y_eval_plot);
    plt::title(ax1, "Target");
    plt::ylabel(ax1, "sin(x)");

    auto ax2 = plt::nexttile();
    plt::plot(ax2, X_eval_plot, Y_pred_plot);
    plt::title(ax2, "Predictions (after training)");
    plt::ylabel(ax2, "forward(x)");
    // plt::ylim(ax2, {-1.0, 1.0});

    auto ax3 = plt::nexttile();
    plt::plot(ax3, epoch_range, loss_history);
    plt::title(ax3, "loss history");
    plt::ylabel(ax3, "loss)");
    plt::xlabel(ax3, "epoch");

    auto ax4 = plt::nexttile();
    plt::plot(ax4, X_eval_plot, Y_pred_pretrained_plot);
    plt::title(ax4, "Predictions (before training)");
    plt::ylabel(ax4, "forward(x)");
    // plt::xlabel(ax3, "epoch");

    plt::show();

    cum::decum();

    return 0;
}
