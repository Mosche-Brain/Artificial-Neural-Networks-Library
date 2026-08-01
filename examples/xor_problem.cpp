#include <iostream>
#include <vector>
#include <functional>
#include <print>

#include <cum/Core.hpp>
#include <cum/cum.hpp>
#include <cum/Matrix.hpp>
#include <cum/runtime.hpp>

#include <yann/models/Sequential.hpp>
#include <yann/utils/FileIO.hpp>
#include <yann/utils/formating.hpp>
#include <yann/runtime_config.hpp>
#include <yann/logging/ITrainingCallback.hpp>
#include <yann/logging/LossTracker.hpp>
#include <yann/optimizers/SGD.hpp>

#include "helpers/surface_visualisation.hpp"
#include "helpers/conversion_helpers.hpp"
#include "loss/BinaryCrossEntropy.hpp"
#include "matplot/axes_objects/surface.h"
#include "matplot/freestanding/plot.h"


int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);
    std::cout << "sizeof cumeric_t in bytes: " << sizeof(cum::cumeric_t) << '\n';
    yann::runtime_config::set_verbosity(1);

    bool plot_before_training = false;
    bool plot_after_training = true;

    yann::models::Sequential sequential({
        yann::models::layers::Input::createUnique(2),
        yann::models::layers::Dense::createUnique(2, "tanh"),
        yann::models::layers::Dense::createUnique(2, "tanh"),
        yann::models::layers::Dense::createUnique(1, "sigmoid")
    });

    // Display weights for each layer
    for (int i = 0 ; i < sequential.getLayersCount() ; i++)
    {
        std::cout << yann::utils::formating::matrixToString(sequential.getWeights(i)) << '\n';
    }



    // cum::cumeric_t min = -5;
    // cum::cumeric_t max = 5;
    // size_t resulution = 100;
    // cum::Matrix samples_batch(2, resulution * resulution);
    // cum::Matrix result_batch(1, resulution * resulution);


    // // linear grid
    // for (int i = 0 ; i < resulution ; i++)
    // {
    //     cum::cumeric_t x = min + (max - min) * i / (resulution - 1);
    //     for (int j = 0 ; j < resulution ; j++)
    //     {
    //         cum::cumeric_t y = min + (max - min) * j / (resulution - 1);
    //         samples_batch(0, i * resulution + j) = x;
    //         samples_batch(1, i * resulution + j) = y;
    //     }
    // }
    //
    // result_batch = sequential.forward(samples_batch);
    //


    cum::Matrix x_train(4, 2,
                       {0, 0,
                        0, 1,
                        1, 0,
                        1, 1});

    cum::Matrix y_train(4, 1, {0, 1, 1, 0});
    cum::Matrix y_eval(4, 1);

    cum::cumeric_t rate = 0.1_c;
    size_t epochs = 500;
    // size_t epochs = 2000;

    for(int i = 0 ; i < 4 ; i++)
    {
        y_eval(i, 0) = sequential.forward(x_train.row(i).transpose())(0,0);
        // cum::runtime::sync();
    }

    std::cout << "pretrain:\n";
    std::cout << '[' << x_train(0, 0) << ',' << x_train(0, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 0) << "\n";
    std::cout << '[' << x_train(1, 0) << ',' << x_train(1, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 1) << "\n";
    std::cout << '[' << x_train(2, 0) << ',' << x_train(2, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 2) << "\n";
    std::cout << '[' << x_train(3, 0) << ',' << x_train(3, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 3) << "\n";

    yann::loss::Loss loss = yann::loss::BinaryCrossEntropy::create();
    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(rate);

    yann::logging::LossTracker loss_tracker = yann::logging::LossTracker();

    std::array<yann::logging::ITrainingCallback*, 1> callbacks = { &loss_tracker };

    sequential.fit(x_train, y_train, *loss, *optimizer, epochs, callbacks);


    std::vector<double> loss_plot;
    std::vector<double> epoch_range;

    for (int i = 0 ; i < loss_tracker.getLossHistory().size() ; i++)
    {
        loss_plot.push_back(loss_tracker.getLossHistory()[i]);
        epoch_range.push_back(i);
    }

    matplot::plot(epoch_range, loss_plot);

    matplot::hold(matplot::off);
    matplot::grid(matplot::on);

    matplot::show();



    y_eval.transposeInPlace();
    x_train.transposeInPlace();

    y_eval = sequential.forward(x_train);
    // for(int i = 0 ; i < 4 ; i++)
        // y_eval(i, 0) = sequential.forward(x_train.row(i).transpose())(0,0);
        // y_eval(i, 0) = sequential.forward(x_train.row(i).transpose())(0,0);


    SurfaceFunc f = [&](cum::cumeric_t x, cum::cumeric_t y) -> cum::cumeric_t
    {
        cum::Matrix sample(2, 1);
        sample(0, 0) = x;
        sample(1, 0) = y;
        cum::runtime::sync();
        cum::cumeric_t result = sequential.forward(sample)(0,0);
        cum::runtime::sync();
        std::println("[{} , {}] -> {}", sample(0,0), sample(1,0), result);
        return result;
    };
    if (plot_after_training)
    {

        //
        auto [X, Y] = matplot::meshgrid(matplot::iota(0, .1, +1));
        auto Z = matplot::transform(X, Y, [=](double x, double y) { return (f(static_cast<cum::cumeric_t>(x), static_cast<cum::cumeric_t>(y))); });
        auto sc = matplot::meshc(X, Y, Z);
        matplot::show();


        // auto* surface = make_function_surface(
        //     f,
        // -5.0f, 5.0f,
        // -5.0f, 5.0f,
        // 150, 150
        // );
        // easy3d::Viewer viewer;
        //
        // viewer.add_model(surface);
        // viewer.run();
    }
    std::cout << "aftertrain:\n";
    for (int i = 0 ; i < 4 ; i++)
        std::cout << '[' << x_train(0, i) << ',' << x_train(1, i) << ']' << ' ' << "->" << ' ' << y_eval(0, i) << "\n";


    for (int i = 0 ; i < sequential.getLayersCount() ; i++)
    {
        std::cout << yann::utils::formating::matrixToString(sequential.getWeights(i)) << '\n';
    }


    cum::decum();

    return 0;
}
