//
// Created by jaro on 6/28/26.
//

#include <matplot/matplot.h>

#include <cum/functions.hpp>

#include <array>


int main()
{
    using namespace matplot;

    fplot([](double x){
        return (double)cum::functions::tanh_derivative((cum::cumeric_t)x);
    }, std::array<double, 2>{-10, 10}, "b");
    hold(on);
    fplot([](double x){
        return (double)cum::functions::tanh((cum::cumeric_t)x);
    }, std::array<double, 2>{-10, 10}, "b");
    hold(off);
    grid(on);

    show();
    return 0;
}