//
// Created by jaro on 8/28/26.
//

#ifndef YANN_HELPERS_HPP
#define YANN_HELPERS_HPP
#include "cum/Core.hpp"
#include "cum/memory.hpp"
#include "cum/functions/various.hpp"

struct TEST_DATA
{
    cum::cumeric_t* source;
    cum::cumeric_t* result_cum;
    cum::cumeric_t* result_std;
    cum::cumeric_t  x_min;
    cum::cumeric_t  x_max;
    size_t N;
};

struct TEST_EVAL
{
    cum::cumeric_t total_error;
    cum::cumeric_t avarage_error;
    cum::cumeric_t max_error;
};



void setup_data(TEST_DATA& data, const cum::cumeric_t min=-10._c, const cum::cumeric_t max=10._c, size_t N=1024)
{
    data.source = cum::memory::allocate(data.N);
    data.result_cum = cum::memory::allocate(data.N);
    data.result_std = cum::memory::allocate(data.N);

    cum::functions::various::linespace(data.source, min, max, N);

    data.x_min = min;
    data.x_max = max;
    data.N = N;
}

void eval_data(TEST_EVAL& results, const TEST_DATA& data)
{
    for (std::size_t i = 0; i < data.N; ++i)
    {
        cum::cumeric_t current_error = std::abs(data.result_cum[i] - data.result_std[i]);
        if (current_error > results.max_error) results.max_error = current_error;
        results.total_error += current_error;
    }
    results.avarage_error = results.total_error / data.N;
}

void print_results(TEST_EVAL& eval)
{
    std::cout << "max error: " << eval.max_error << std::endl;
    std::cout << "avarage error: " << eval.avarage_error << std::endl;
}

#endif //YANN_HELPERS_HPP
