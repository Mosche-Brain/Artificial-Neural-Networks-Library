//
// Created by jaro on 7/31/26.
//

#include <yann/models/Sequential.hpp>
#include <cum/cum.hpp>

#include <time.h>

#include <print>

#include <matplot/matplot.h>

namespace plt = matplot;

int main()
{
    cum::cum(cum::CUM_DEVICE::GPU);


    yann::models::Sequential sequential({
        yann::models::layers::Input::createUnique(8),
        yann::models::layers::Dense::createUnique(16, "tanh"),
        yann::models::layers::Dense::createUnique(16, "tanh"),
        yann::models::layers::Dense::createUnique(16, "tanh"),
        yann::models::layers::Dense::createUnique(16, "tanh"),
        yann::models::layers::Dense::createUnique(16, "tanh"),
        yann::models::layers::Dense::createUnique(16, "tanh"),
        yann::models::layers::Dense::createUnique(1, "sigmoid")
    });


    size_t samples_count = 128;
    size_t input_size = 8;

    cum::Matrix batched_samples = cum::Matrix::Random(input_size, samples_count);

    std::vector<cum::Matrix> samples(samples_count);
    for (size_t i = 0; i < samples_count; i++)
    {
        samples[i] = (cum::Matrix::Random(input_size, 1));
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    sequential.forward(batched_samples);

    clock_gettime(CLOCK_MONOTONIC, &end);
    long long elapsed = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    double time1 = elapsed / 1000000000.0;

    std::println("Time with batches: {}s", time1);

    clock_gettime(CLOCK_MONOTONIC, &start);

    for (auto& sample : samples)
        sequential.forward(sample);

    sequential.forward(batched_samples);

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    double time2 = elapsed / 1000000000.0;

    std::println("Time without batches: {}s", time2);

    std::vector<double> y = {time1, time2};
    plt::bar(y);
    plt::gca()->x_axis().ticklabels({"batche", "pojedyńcze"});


    plt::show();

    cum::decum();
}
