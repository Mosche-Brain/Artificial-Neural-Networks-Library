//
// Created by jaro on 8/28/26.
//
#include <catch2/catch_test_macros.hpp>

#include <cum/cum.hpp>
#include <cum/functions.hpp>
#include <cum/memory.hpp>

#include <cmath>
#include <functional>

#include "cum/runtime.hpp"

#include "helpers.hpp"

// cum::cumeric_t diff(cum::cumeric_t* dY, const cum::cumeric_t* Y, const cum::cumeric_t* Y_0, const cum::cumeric_t* X, const cum::cumeric_t* X_0);
cum::cumeric_t diff(cum::cumeric_t* dY, const cum::cumeric_t* X, std::function<cum::cumeric_t(cum::cumeric_t)> f, const size_t N, const cum::cumeric_t h=1e-3_c)
{
    for (size_t i = 0; i < N; ++i)
    {
        cum::cumeric_t y1 = f(X[i] + h);
        cum::cumeric_t y = f(X[i]);
        dY[i] = (y1 - y) / h;
    }
}

TEST_CASE("tanh'")
{
    cum::cum(cum::DEVICE::CPU);

    TEST_DATA data;
    setup_data(data);

    for (std::size_t i = 0; i < data.N; ++i)
    {
        data.result_std[i] = 1 - std::tanh(data.source[i]) * std::tanh(data.source[i]);
    }
    // diff(data.result_std, data.source, std::tanhf, data.N);
    cum::functions::hyperbolic::tanh_deriv(data.result_cum, data.source, data.N);
    cum::runtime::sync();
    std::cout << data.result_std[data.N/2] << std::endl;
    std::cout << data.result_cum[data.N/2] << std::endl;

    TEST_EVAL eval{};
    eval_data(eval, data);
    print_results(eval);

    cum::decum();
    REQUIRE(eval.avarage_error < 1e-2_c);
}
