//
// Created by jaro on 7/17/26.
//
#include <math.h>

#include "Window.hpp"

#include <YANN/Models/Sequential.hpp>

#include "runtime_config.hpp"

int main()
{
    Window::InitBackend();
    Window window(1920, 1080, "Brainscape");


    cum::cum(cum::CUM_DEVICE::CPU);

    yann::runtime_config::set_verbosity(5);

    yann::models::Sequential model({
        yann::models::layers::Input::createUnique(1),
        yann::models::layers::Dense::createUnique(32, "tanh"),
        // yann::models::layers::Dense::createUnique(32, "tanh"),
        yann::models::layers::Dense::createUnique(1, "tanh"),
    });

    cum::cumeric_t x_min = -8.0 * M_PIf;
    cum::cumeric_t x_max =  8.0 * M_PIf;
    std::size_t N_train = 8;
    std::size_t N_eval = 512;

    cum::Matrix X_train = cum::Matrix::Linspace(x_min, x_max, N_train).transpose();
    cum::Matrix X_eval = cum::Matrix::Linspace(x_min, x_max, N_eval).transpose();
    cum::Matrix Y_train = cum::Matrix::Linspace(x_min, x_max, N_train).transpose();
    cum::Matrix Y_eval = cum::Matrix::Linspace(x_min, x_max, N_eval);

    cum::LinearAlgebra::sinInPlace(Y_train.data(), N_train);

    while (!window.shouldClose())
    {

        window.pollEvents();
        window.swapBuffers();
    }

    cum::decum();
    return 0;
}
