//
// Created by jaro on 7/5/26.
//

#include <yann/models/Sequential.hpp>
#include <cum/cum.hpp>
#include <cum/random.hpp>

#include <cmath>

#include "runtime_config.hpp"
#include "helpers/surface_visualisation.hpp"

using cum::cumeric_t;
using cum::Matrix;

cumeric_t f(cumeric_t x, cumeric_t y)
{
    return sin(sqrt(x * x + y * y));
}

int main()
{
    cum::cum(          cum::DEVICE::CPU);
    yann::models::Sequential model({
        yann::models::layers::Input::createUnique(2),
        // yann::models::layers::Dense::createUnique(256, "relu"),
        yann::models::layers::Dense::createUnique(128, "tanh"),
        yann::models::layers::Dense::createUnique(1, "tanh")
    });

    yann::runtime_config::set_verbosity(5);

    cumeric_t x_from = -4*M_PI;
    cumeric_t y_from = -4*M_PI;
    cumeric_t x_to = 4*M_PI;
    cumeric_t y_to = 4*M_PI;

    SurfaceFunc model_forward = [&](cum::cumeric_t x, cum::cumeric_t y) -> cum::cumeric_t {
        cum::Matrix sample(2, 1);
        sample(0, 0) = x;
        sample(1, 0) = y;
        return model.forward(sample)(0,0);
    };

    Matrix XY(200, 2);
    cum::random::uniform(XY.data(), XY.rows() * XY.cols(), x_from, x_to);

    Matrix Z(200, 1);
    for (int i = 0; i < Z.rows(); ++i)
    {
        Z(i, 0) = f(XY(i,0), XY(i,1));
    }

    cumeric_t rate = 0.1;
    size_t epochs = 1;
    model.setLossFunction(yann::utils::loss::LossFunction::mse);
    model.fit(XY, Z, rate, epochs);

    return 0;

    auto* surface = make_function_surface(
        model_forward,
        2*x_from, 2*x_to,
        2*y_from, 2*y_to,
        50, 50
    );

    easy3d::initialize();
    easy3d::Viewer viewer;
    viewer.add_model(surface);
    viewer.run();

    cum::decum();
    return 0;
}