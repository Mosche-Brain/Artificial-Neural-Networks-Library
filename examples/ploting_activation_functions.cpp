//
// Created by jaro on 6/28/26.
//

#include <matplot/matplot.h>

#include <cum/functions.hpp>
#include <cum/LinearAlgebra.hpp>

#include <array>

#include <cum/Matrix.hpp>
#include <cum/memory.hpp>

#include "helpers/conversion_helpers.hpp"

int main()
{
    using namespace matplot;

    // fplot([](double x){
        // return (double)cum::functions::tanh_derivative((cum::cumeric_t)x);
    // }, std::array<double, 2>{-10, 10}, "b");
    // hold(on);

    cum::Matrix X = cum::Matrix::Linspace(-10, 10, 100);
    cum::Matrix Y = cum::Matrix::Linspace(-10, 10, 100);

    cum::functions::linear(Y.data(), X.data(), 100);
    // cum::functions::tanh_derivative(Y.data(), X.data(), 100);
    // cum::functions::tanh(Y.data(), X.data(), 100);

    std::vector<double> X_plot = toStdVector<double>(X);
    std::vector<double> Y_plot = toStdVector<double>(Y);

    plot(X_plot, Y_plot);

    // fplot([](double x){
    //     // return (double)cum::functions::tanh((cum::cumeric_t)x);
    //     cum::cumeric_t* temp = cum::memory::allocate(1);
    //     cum::cumeric_t* rslt = cum::memory::allocate(1);
    //     temp[0] = (cum::cumeric_t)x;
    //     cum::functions::tanh_derivative(rslt, temp, 1);
    //     cum::cumeric_t y = rslt[0];
    //     cum::memory::free(temp);
    //     cum::memory::free(rslt);
    //     return (double)y;
    // }, std::array<double, 2>{-10, 10}, "b");
    hold(off);
    grid(on);

    show();
    return 0;
}
