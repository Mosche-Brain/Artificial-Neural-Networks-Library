/*
 * @author: jaro
 * @name:   backpropagation
 * @file:   tests/backpropagation.cpp
 */

#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cum/cum.hpp>

#include <yann/models/layers/Dense.hpp>
#include <yann/models/layers/Linear.hpp>

namespace
{
    void set_values(cum::Tensor& tensor, std::initializer_list<cum::cumeric_t> values)
    {
        REQUIRE(tensor.lenght() == static_cast<cum::dim_t>(values.size()));
        auto value = values.begin();
        for (cum::dim_t row = 0; row < tensor.rows(); ++row)
            for (cum::dim_t col = 0; col < tensor.cols(); ++col)
                tensor.at<cum::cumeric_t>({row, col}) = *value++;
    }

    void require_values(const cum::Tensor& tensor, std::initializer_list<cum::cumeric_t> expected)
    {
        REQUIRE(tensor.lenght() == static_cast<cum::dim_t>(expected.size()));
        auto value = expected.begin();
        for (cum::dim_t row = 0; row < tensor.rows(); ++row)
            for (cum::dim_t col = 0; col < tensor.cols(); ++col)
                REQUIRE_THAT(tensor.at<cum::cumeric_t>({row, col}),
                             Catch::Matchers::WithinAbs(*value++, static_cast<cum::cumeric_t>(1e-5)));
    }
    std::vector<yann::Parameter*> set_linear_parameters(yann::models::layers::LayerBase& layer)
    {
        std::vector<yann::Parameter*> parameters;
        layer.collect_parameters(parameters);
        REQUIRE(parameters.size() == 2);
        set_values(parameters[0]->values, {1, -2, 0.5, 3});
        set_values(parameters[1]->values, {0.25, -0.5});
        return parameters;
    }
}

TEST_CASE("Linear backward computes input and parameter gradients")
{
    cum::cum(cum::DEVICE::CPU);

    yann::models::layers::Linear layer(2);
    layer.init_parameters(2, 2);
    const auto parameters = set_linear_parameters(layer);

    cum::Tensor input({2, 2}, cum::default_type, cum::layout::IO);
    set_values(input, {2, -1, 1, 4});
    layer.forward(input);

    cum::Tensor output_gradient({2, 2}, cum::default_type, cum::layout::IO);
    set_values(output_gradient, {2, -1, 3, 4});
    const cum::Tensor input_gradient = layer.backward(output_gradient);

    require_values(input_gradient, {3.5, 1, 5, 14});
    require_values(parameters[0]->gradient, {5, -2, 2, 19});
    require_values(parameters[1]->gradient, {1, 7});
    cum::decum();
}

TEST_CASE("Dense backward includes the ReLU derivative")
{
    cum::cum(cum::DEVICE::CPU);

    yann::models::layers::Dense layer(2, "relu");
    layer.init_parameters(2, 2);
    const auto parameters = set_linear_parameters(layer);

    cum::Tensor input({2, 2}, cum::default_type, cum::layout::IO);
    set_values(input, {2, -1, 1, 4});
    layer.forward(input);

    cum::Tensor output_gradient({2, 2}, cum::default_type, cum::layout::IO);
    set_values(output_gradient, {2, -1, 3, 4});
    const cum::Tensor input_gradient = layer.backward(output_gradient);

    require_values(input_gradient, {3.5, 2, 5, 12});
    require_values(parameters[0]->gradient, {4, 2, 2, 19});
    require_values(parameters[1]->gradient, {2, 7});
    cum::decum();
}

