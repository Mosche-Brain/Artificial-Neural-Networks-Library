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

#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cum/Tensor.hpp>
#include <cum/neural_primitives/EltwiseKernel.hpp>

void require_eltwise_gradient(cum::functions::function_id algorithm, const std::function<cum::cumeric_t(cum::cumeric_t)>& function)
{
    constexpr cum::dim_t count = 4;

    const cum::cumeric_t h = static_cast<cum::cumeric_t>(1e-3);

    cum::Tensor input({count}, cum::default_type, cum::layout::X);

    cum::cumeric_t* input_data = static_cast<cum::cumeric_t*>(input.data());

    input_data[0] = static_cast<cum::cumeric_t>(-0.8);
    input_data[1] = static_cast<cum::cumeric_t>(-0.2);
    input_data[2] = static_cast<cum::cumeric_t>(0.3);
    input_data[3] = static_cast<cum::cumeric_t>(0.9);

    cum::neural_primitives::EltwiseKernel kernel(*input.descriptor(), algorithm);
    cum::cumeric_t numerical[count];
    for(cum::dim_t i = 0; i < count; ++i)
        numerical[i] = (function(input_data[i] + h) - function(input_data[i] - h)) / (2 * h);

    cum::Tensor gradient({count}, cum::default_type, cum::layout::X);
    gradient.fill(1);
    kernel.backward(gradient, input);

    cum::cumeric_t* gradient_data = static_cast<cum::cumeric_t*>(gradient.data());
    for(cum::dim_t i = 0; i < count; ++i)
        REQUIRE_THAT(gradient_data[i],
                     Catch::Matchers::WithinRel(numerical[i], static_cast<cum::cumeric_t>(5e-3)));
}


TEST_CASE("EltwiseKernel tanh backward matches central numerical differentiation")
{
    cum::cum(cum::DEVICE::CPU);
    require_eltwise_gradient(cum::functions::function_id::tanh,
                             [](cum::cumeric_t value) { return std::tanh(value); });
    cum::decum();
}

TEST_CASE("EltwiseKernel sigmoid backward matches central numerical differentiation")
{
    cum::cum(cum::DEVICE::CPU);
    require_eltwise_gradient(cum::functions::function_id::sigmoid,
                             [](cum::cumeric_t value) {
                                 return 1 / (1 + std::exp(-value));
                             });
    cum::decum();
}

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
