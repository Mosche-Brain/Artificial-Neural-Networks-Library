#include <iostream>
#include <vector>
#include <functional>

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
#include "matplot/axes_objects/surface.h"
#include "matplot/freestanding/plot.h"


#include <unistd.h>

int main()
{
    cum::cum(cum::CUM_DEVICE::CPU);
    std::cout << "sizeof cumeric_t in bytes: " << sizeof(cum::cumeric_t) << '\n';
    yann::runtime_config::set_verbosity(0);


    yann::models::Sequential sequential({
        yann::models::layers::Input::createUnique(2),
        yann::models::layers::Dense::createUnique(2, "tanh"),
        yann::models::layers::Dense::createUnique(1, "linear")
    });

    // Display weights for each layer
    for (int i = 0 ; i < sequential.getLayersCount() ; i++)
    {
        std::cout << yann::utils::formating::matrixToString(sequential.getWeights(i)) << '\n';
    }


    SurfaceFunc f = [&](cum::cumeric_t x, cum::cumeric_t y) -> cum::cumeric_t
    {
        // usleep(100);
        cum::Matrix sample(2, 1);
        sample(0, 0) = x;
        sample(1, 0) = y;
        // cum::runtime::sync();
        return sequential.forward(sample)(0,0);
    };
    //
    auto* surface1 = make_function_surface(
    f,
    -5.0f, 5.0f,
    -5.0f, 5.0f,
    150, 150
    );



    cum::Matrix x_train(4, 2,
                       {0, 0,
                        0, 1,
                        1, 0,
                        1, 1});

    cum::Matrix y_train(4, 1, {0, 1, 1, 0});
    cum::Matrix y_eval(4, 1);

    cum::cumeric_t rate = 0.1_c;
    size_t epochs = 3000;
    // size_t epochs = 2500;

    for(int i = 0 ; i < 4 ; i++)
    {
        y_eval(i, 0) = sequential.forward(x_train.row(i).transpose())(0,0);
        cum::runtime::sync();
    }

    std::cout << "pretrain:\n";
    std::cout << '[' << x_train(0, 0) << ',' << x_train(0, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 0) << "\n";
    std::cout << '[' << x_train(1, 0) << ',' << x_train(1, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 1) << "\n";
    std::cout << '[' << x_train(2, 0) << ',' << x_train(2, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 2) << "\n";
    std::cout << '[' << x_train(3, 0) << ',' << x_train(3, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 3) << "\n";




    auto [X, Y] = matplot::meshgrid(matplot::iota(-5, .2, +5));

    auto Z = matplot::transform(
            X, Y, [=](double x, double y) { return static_cast<double>(f(static_cast<cum::cumeric_t>(x), static_cast<cum::cumeric_t>(y))); });


    // return 0;
    auto sc = matplot::meshc(X, Y, Z);
    // sc->edge_color("r");
    // sc->contour_line_spec().color("b");



    matplot::show();

    cum::decum();




    return 0;

    sequential.setLossFunction(yann::loss::LossFunction::binary_cross_entropy);


    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(rate);

    yann::logging::LossTracker loss_tracker = yann::logging::LossTracker();

    std::array<yann::logging::ITrainingCallback*, 1> callbacks = { &loss_tracker };

    std::getchar();
    sequential.fit(x_train, y_train, *optimizer, epochs, callbacks);

    cum::Matrix sample(1, 2, 1._c);




    std::vector<double> loss;
    std::vector<double> epoch_range;

    for (int i = 0 ; i < loss_tracker.getLossHistory().size() ; i++)
    {
        loss.push_back(loss_tracker.getLossHistory()[i]);
        epoch_range.push_back(i);
    }

    matplot::plot(epoch_range, loss);

    matplot::hold(matplot::off);
    matplot::grid(matplot::on);

    matplot::show();
    //

    for(int i = 0 ; i < 4 ; i++)
        y_eval(i, 0) = sequential.forward(x_train.row(i).transpose())(0,0);

    std::cout << "aftertrain:\n";
    std::cout << '[' << x_train(0, 0) << ',' << x_train(0, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 0) << "\n";
    std::cout << '[' << x_train(1, 0) << ',' << x_train(1, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 1) << "\n";
    std::cout << '[' << x_train(2, 0) << ',' << x_train(2, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 2) << "\n";
    std::cout << '[' << x_train(3, 0) << ',' << x_train(3, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 3) << "\n";


    for (int i = 0 ; i < sequential.getLayersCount() ; i++)
    {
        std::cout << yann::utils::formating::matrixToString(sequential.getWeights(i)) << '\n';
    }



    auto* surface2 = make_function_surface(
    f,
    -5.0f, 5.0f,
    -5.0f, 5.0f,
    150, 150
    );


    cum::decum();

    return 0;
}
