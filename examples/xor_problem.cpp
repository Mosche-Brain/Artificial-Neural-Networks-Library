#include <iostream>
#include <vector>
#include <functional>

#include <cum/Core.hpp>
#include <cum/cum.hpp>
#include <cum/Matrix.hpp>

#include <YANN/Models/Sequential.hpp>
#include <YANN/Utility/FileIO.hpp>
#include <YANN/runtime_config.hpp>

#include "helpers/surface_visualisation.hpp"

int main()
{
    cum::cum(cum::CUM_DEVICE::CPU);
    std::cout << "sizeof cumeric_t in bytes: " << sizeof(cum::cumeric_t) << '\n';
    yann::runtime_config::set_verbosity(0);


    easy3d::initialize();
    easy3d::Viewer viewer("plot");

    yann::models::Sequential sequential({
        yann::models::layers::Input::createUnique(2),
        yann::models::layers::Dense::createUnique(3, "sigmoid"),
        yann::models::layers::Dense::createUnique(1, "sigmoid")
    });


    SurfaceFunc f = [&](cum::cumeric_t x, cum::cumeric_t y) -> cum::cumeric_t {
        cum::Matrix sample(2, 1);
        sample(0, 0) = x;
        sample(1, 0) = y;
        return sequential.forward(sample)(0,0);
    };

    auto* surface1 = make_function_surface(
    f,
    -5.0f, 5.0f,
    -5.0f, 5.0f,
    150, 150
    );

    viewer.add_model(surface1, false);

    cum::Matrix x_train(4, 2,
                       {0, 0,
                        0, 1,
                        1, 0,
                        1, 1});

    cum::Matrix y_train(4, 1, {0, 1, 1, 0});
    cum::Matrix y_eval(4, 1);

    cum::cumeric_t rate = 0.1_c;
    size_t epochs = 3000;

    for(int i = 0 ; i < 4 ; i++)
        y_eval(i, 0) = sequential.forward(x_train.row(i).transpose())(0,0);

    std::cout << "pretrain:\n";
    std::cout << '[' << x_train(0, 0) << ',' << x_train(0, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 0) << "\n";
    std::cout << '[' << x_train(1, 0) << ',' << x_train(1, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 1) << "\n";
    std::cout << '[' << x_train(2, 0) << ',' << x_train(2, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 2) << "\n";
    std::cout << '[' << x_train(3, 0) << ',' << x_train(3, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 3) << "\n";

    sequential.setLossFunction(yann::Utils::loss::LossFunction::binary_cross_entropy);

    sequential.fit(x_train, y_train, rate, epochs);

    cum::Matrix sample(1, 2, 1._c);


    for(int i = 0 ; i < 4 ; i++)
        y_eval(i, 0) = sequential.forward(x_train.row(i).transpose())(0,0);

    std::cout << "aftertrain:\n";
    std::cout << '[' << x_train(0, 0) << ',' << x_train(0, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 0) << "\n";
    std::cout << '[' << x_train(1, 0) << ',' << x_train(1, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 1) << "\n";
    std::cout << '[' << x_train(2, 0) << ',' << x_train(2, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 2) << "\n";
    std::cout << '[' << x_train(3, 0) << ',' << x_train(3, 1) << ']' << ' ' << "->" << ' ' << y_eval(0, 3) << "\n";



    auto* surface2 = make_function_surface(
    f,
    -5.0f, 5.0f,
    -5.0f, 5.0f,
    150, 150
    );

    viewer.add_model(surface2, true);

    viewer.run();
    cum::decum();

    return 0;
}
