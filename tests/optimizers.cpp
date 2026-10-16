 /*
 * @name:   optimizers
 * @file:   tests/optimizers.cpp
 */

#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cum/cum.hpp>

#include <yann/Parameter.hpp>
#include <yann/optimizers/ADAM.hpp>

namespace
{
    constexpr double tolerance = 1e-5;

    void set_row(cum::Tensor& tensor, const std::vector<double>& values)
    {
        for (cum::dim_t col = 0; col < static_cast<cum::dim_t>(values.size()); ++col)
            tensor.at<cum::cumeric_t>({0, col}) = static_cast<cum::cumeric_t>(values[col]);
    }

    double get(const cum::Tensor& tensor, cum::dim_t col)
    {
        return static_cast<double>(tensor.at<cum::cumeric_t>({0, col}));
    }

    // Plain scalar implementation of Adam (Kingma & Ba, 2015), independent of cum.
    struct ReferenceAdam
    {
        double lr, b1, b2, eps;
        std::vector<double> m, v;
        int t = 0;

        ReferenceAdam(double lr, double b1, double b2, std::size_t size)
            : lr(lr), b1(b1), b2(b2), eps(static_cast<double>(cum::EPSILON)), m(size, 0.0), v(size, 0.0)
        {}

        void step(std::vector<double>& w, const std::vector<double>& g)
        {
            ++t;
            for (std::size_t i = 0; i < w.size(); ++i)
            {
                m[i] = b1 * m[i] + (1 - b1) * g[i];
                v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];

                const double m_hat = m[i] / (1 - std::pow(b1, t));
                const double v_hat = v[i] / (1 - std::pow(b2, t));

                w[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    };
}

TEST_CASE("ADAM first step moves each weight by learning_rate in the direction opposite to the gradient")
{
    cum::cum(cum::DEVICE::CPU);

    yann::Parameter param = yann::Parameter::Zeros(1, 2);
    set_row(param.values, {1.0, -1.0});
    set_row(param.gradient, {0.5, -2.0});

    yann::optimizers::ADAM adam(0.1);
    std::vector<yann::Parameter*> params = {&param};
    adam.step(params);

    // With bias correction: m_hat = g, v_hat = g^2, so the update is lr * g / (|g| + eps) ~= lr * sign(g)
    REQUIRE_THAT(get(param.values, 0), Catch::Matchers::WithinAbs(0.9, tolerance));
    REQUIRE_THAT(get(param.values, 1), Catch::Matchers::WithinAbs(-0.9, tolerance));

    cum::decum();
}

TEST_CASE("ADAM clears gradients after a step")
{
    cum::cum(cum::DEVICE::CPU);

    yann::Parameter param = yann::Parameter::Zeros(1, 2);
    set_row(param.gradient, {0.5, -2.0});

    yann::optimizers::ADAM adam(0.1);
    std::vector<yann::Parameter*> params = {&param};
    adam.step(params);

    REQUIRE_THAT(get(param.gradient, 0), Catch::Matchers::WithinAbs(0.0, tolerance));
    REQUIRE_THAT(get(param.gradient, 1), Catch::Matchers::WithinAbs(0.0, tolerance));

    cum::decum();
}

TEST_CASE("ADAM matches a scalar reference implementation over several steps")
{
    cum::cum(cum::DEVICE::CPU);

    constexpr double learning_rate = 0.05;
    constexpr double b1 = 0.9;
    constexpr double b2 = 0.999;

    const std::vector<std::vector<double>> gradients = {
        { 0.5, -2.0,  0.0},
        {-0.5,  1.0,  0.25},
        { 1.5,  0.1, -0.75},
        { 0.2, -0.3,  0.4},
        {-1.0,  2.0,  0.1},
    };

    std::vector<double> reference_weights = {1.0, -1.0, 0.5};
    ReferenceAdam reference(learning_rate, b1, b2, reference_weights.size());

    yann::Parameter param = yann::Parameter::Zeros(1, 3);
    set_row(param.values, reference_weights);

    yann::optimizers::ADAM adam(learning_rate, b1, b2);
    std::vector<yann::Parameter*> params = {&param};

    for (const std::vector<double>& gradient : gradients)
    {
        set_row(param.gradient, gradient);
        adam.step(params);

        reference.step(reference_weights, gradient);

        for (cum::dim_t col = 0; col < 3; ++col)
            REQUIRE_THAT(get(param.values, col), Catch::Matchers::WithinAbs(reference_weights[col], tolerance));
    }

    cum::decum();
}

TEST_CASE("ADAM keeps independent state for every parameter")
{
    cum::cum(cum::DEVICE::CPU);

    yann::Parameter a = yann::Parameter::Zeros(1, 1);
    yann::Parameter b = yann::Parameter::Zeros(1, 1);
    set_row(a.values, {1.0});
    set_row(b.values, {1.0});

    ReferenceAdam reference_a(0.1, 0.9, 0.999, 1);
    ReferenceAdam reference_b(0.1, 0.9, 0.999, 1);
    std::vector<double> wa = {1.0};
    std::vector<double> wb = {1.0};

    yann::optimizers::ADAM adam(0.1);
    std::vector<yann::Parameter*> params = {&a, &b};

    for (int step = 0; step < 3; ++step)
    {
        const std::vector<double> ga = {0.3 * (step + 1)};
        const std::vector<double> gb = {-5.0 / (step + 1)};

        set_row(a.gradient, ga);
        set_row(b.gradient, gb);
        adam.step(params);

        reference_a.step(wa, ga);
        reference_b.step(wb, gb);

        REQUIRE_THAT(get(a.values, 0), Catch::Matchers::WithinAbs(wa[0], tolerance));
        REQUIRE_THAT(get(b.values, 0), Catch::Matchers::WithinAbs(wb[0], tolerance));
    }

    cum::decum();
}
