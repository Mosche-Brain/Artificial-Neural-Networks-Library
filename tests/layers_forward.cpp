/*
 * @author: jaro
 * @name:   layers_forward
 * @file:   tests/layers_forward.cpp
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
    void set_linear_parameters(yann::models::layers::LayerBase& layer)
    {
        std::vector<yann::Parameter*> parameters;
        layer.collect_parameters(parameters);
        REQUIRE(parameters.size() == 2);
        set_values(parameters[0]->values, {1, -2, 0.5, 3});
        set_values(parameters[1]->values, {0.25, -0.5});
    }
}

TEST_CASE("Linear forward computes affine transformation")
{
    cum::cum(cum::DEVICE::CPU);

    yann::models::layers::Linear layer(2);
    layer.init_parameters(2, 2);
    set_linear_parameters(layer);

    cum::Tensor input({2, 2}, cum::default_type, cum::layout::IO);
    set_values(input, {2, -1, 1, 4});

    const cum::Tensor output = layer.forward(input);

    REQUIRE(output.shape() == (cum::Shape{2, 2}));
    require_values(output, {0.25, -8.75, 3.5, 11});
    cum::decum();
}

TEST_CASE("Dense forward applies affine transformation and ReLU")
{
    cum::cum(cum::DEVICE::CPU);

    yann::models::layers::Dense layer(2, "relu");
    layer.init_parameters(2, 2);
    set_linear_parameters(layer);

    cum::Tensor input({2, 2}, cum::default_type, cum::layout::IO);
    set_values(input, {2, -1, 1, 4});

    const cum::Tensor output = layer.forward(input);

    REQUIRE(output.shape() == (cum::Shape{2, 2}));
    require_values(output, {0.25, 0, 3.5, 11});
}