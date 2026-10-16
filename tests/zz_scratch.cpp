#include <cstdio>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <cum/cum.hpp>
#include <cum/Tensor.hpp>
#include <yann/Parameter.hpp>
#include <yann/optimizers/ADAM.hpp>

static void run(cum::dim_t cols, float g0)
{
    yann::Parameter p = yann::Parameter::Zeros(1, cols);
    for (cum::dim_t c = 0; c < cols; ++c)
    {
        p.values.at<cum::cumeric_t>({0, c}) = 1.0f;
        p.gradient.at<cum::cumeric_t>({0, c}) = g0;
    }
    yann::optimizers::ADAM adam(0.1);
    std::vector<yann::Parameter*> v = {&p};
    adam.step(v);
    std::printf("cols=%ld g=%g -> w0=%.9g\n", (long)cols, g0, p.values.at<cum::cumeric_t>({0, 0}));
}

TEST_CASE("scratch")
{
    cum::cum(cum::DEVICE::CPU);
    run(1, 0.3f);
    run(1, 0.5f);
    run(2, 0.3f);
    run(3, 0.3f);
    run(1, -5.0f);
    run(1, 2.0f);
    cum::decum();
}
