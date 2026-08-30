#include <cum/cum.hpp>
#include <cum/Matrix.hpp>
#include <cum/runtime.hpp>

#include <yann/models/Sequential.hpp>
#include <yann/optimizers/SGD.hpp>
#include <yann/optimizers/ADAM.hpp>
#include <yann/loss/MeanSquaredError.hpp>
#include <yann/logging/LossTracker.hpp>
#include <yann/logging/GradientTracker.hpp>
#include <yann/runtime_config.hpp>

#include <matplot/matplot.h>

#include <iostream>
#include <numeric>

#include "helpers/conversion_helpers.hpp"

namespace plt = matplot;

int main()
{
    cum::cum(cum::CUM_DEVICE::CPU);

    yann::runtime_config::set_verbosity(0);

    // yann::models::Sequential model({
    //     yann::models::layers::Input::createUnique(1),
    //     // yann::models::layers::Dense::createUnique(128, "relu"),
    //     yann::models::layers::Dense::createUnique(128, "relu"),
    //     yann::models::layers::Dense::createUnique(1, "tanh"),
    // });

    yann::models::Sequential model;
    model.addLayer(yann::models::layers::Input::createUnique(1));
    model.addLayer(yann::models::layers::Dense::createUnique(128, "relu"));
    model.addLayer(yann::models::layers::Dense::createUnique(128, "relu"));
    model.addLayer(yann::models::layers::Dense::createUnique(1, "tanh"));

    model.build();

    cum::cumeric_t x_min = 4.f * -M_PIf;
    cum::cumeric_t x_max = 4.f * M_PIf;
    std::size_t N_train = 64;
    std::size_t N_eval = 512;

    cum::Matrix X_train = cum::Matrix::Linspace(x_min, x_max, N_train).transpose(); // linspace in row vector
    cum::Matrix Y_train = cum::Matrix::Linspace(x_min, x_max, N_train).transpose();

    cum::Matrix X_eval = cum::Matrix::Linspace(x_min, x_max, N_eval);
    cum::Matrix Y_eval = cum::Matrix::Linspace(x_min, x_max, N_eval);

    cum::functions::trigonometric::sin_in_place(Y_eval.data(), N_eval);
    cum::functions::trigonometric::sin_in_place(Y_train.data(), N_train);

    //X_train /= x_max;
    //X_eval /= x_max;

    cum::Matrix Y_pred_pretrained = model.forward(X_eval); // batch

    yann::loss::Loss loss = yann::loss::MeanSquaredError::create();
    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(0.001);


    yann::logging::LossTracker loss_tracker = yann::logging::LossTracker();
    yann::logging::GradientTracker grad_tracker = yann::logging::GradientTracker();

    std::array<yann::logging::ITrainingCallback*, 2> callbacks = { &loss_tracker, &grad_tracker };

    yann::runtime_config::set_verbosity(1);
    model.fit(X_train, Y_train, *loss, *optimizer, 8000, 64, callbacks);

    std::vector<double> loss_history;
    std::vector<double> epoch_range;

    for (int i = 0 ; i < loss_tracker.getLossHistory().size() ; i++)
    {
        loss_history.push_back(loss_tracker.getLossHistory()[i]);
        epoch_range.push_back(i);
    }

    cum::Matrix Y_pred = model.forward(X_eval);

    /* Plotting results */
    std::vector<double> X_train_plot = toStdVector<double>(X_train);
    std::vector<double> Y_train_plot = toStdVector<double>(Y_train);

    std::vector<double> X_eval_plot = toStdVector<double>(X_eval);
    std::vector<double> Y_eval_plot = toStdVector<double>(Y_eval);
    std::vector<double> Y_pred_plot = toStdVector<double>(Y_pred);
    std::vector<double> Y_pred_pretrained_plot = toStdVector<double>(Y_pred_pretrained);

    plt::gcf()->size(1400, 900);

    plt::tiledlayout(2, 3);
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

    auto ax5 = plt::nexttile();
    plt::hold(ax5, plt::on);
    const auto gradient_history = grad_tracker.getLayersHistory();
    for (std::size_t layer = 1; layer < gradient_history.size(); ++layer)
    {
        const auto& layer_history = gradient_history[layer];
        std::vector<double> batch_range(layer_history.weights.size());
        std::iota(batch_range.begin(), batch_range.end(), 0.0);

        auto weights_line = plt::plot(ax5, batch_range, layer_history.weights);
        weights_line->display_name("layer " + std::to_string(layer) + " weights");
        auto biases_line = plt::plot(ax5, batch_range, layer_history.biases);
        biases_line->display_name("layer " + std::to_string(layer) + " biases");
    }
    plt::hold(ax5, plt::off);
    plt::title(ax5, "Average absolute gradient size");
    plt::ylabel(ax5, "mean(|gradient|)");
    plt::xlabel(ax5, "batch");
    plt::legend(ax5);

    plt::show();

    cum::decum();

    return 0;
}
