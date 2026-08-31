//
// Created by jaro on 6/28/26.
//

#include <matplot/matplot.h>

#include <cum/functions.hpp>
#include <cum/LinearAlgebra.hpp>

#include <array>

#include <cum/Matrix.hpp>
#include <cum/memory.hpp>

#include "cum/cum.hpp"
#include "cum/runtime.hpp"
#include "helpers/conversion_helpers.hpp"

int main()
{
    using namespace matplot;

    cum::cum(cum::CUM_DEVICE::CPU);
    // fplot([](double x){
        // return (double)cum::functions::tanh_derivative((cum::cumeric_t)x);
    // }, std::array<double, 2>{-10, 10}, "b");
    // hold(on);

    cum::Matrix X = cum::Matrix::Linspace(-10, 10, 100);
    cum::Matrix Y = cum::Matrix::Linspace(-10, 10, 100);

    cum::functions::logistic::sigmoid_deriv(Y.data(), X.data(), 100);
    cum::runtime::sync();
    // cum::functions::tanh_derivative(Y.data(), X.data(), 100);
    // cum::functions::tanh(Y.data(), X.data(), 100);

    std::vector<double> X_plot = toStdVector<double>(X);
    std::vector<double> Y_plot = toStdVector<double>(Y);

    plot(X_plot, Y_plot);

    hold(off);
    grid(on);

    show();
    return 0;
}
