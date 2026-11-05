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
#include <yann/models/layers/Input.hpp>
#include <yann/models/layers/Flatten.hpp>
#include <yann/models/Sequential.hpp>

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
        set_values(parameters[0]->values, {1, 0.5, -2, 3} /* [in, out] */);
        set_values(parameters[1]->values, {0.25, -0.5});
    }
}

TEST_CASE("Linear forward computes affine transformation")
{
    cum::cum(cum::DEVICE::CPU);

    yann::models::layers::Linear layer(2);
    layer.init_parameters(2, 2);
    set_linear_parameters(layer);

    cum::Tensor input({2, 2}, cum::default_type, cum::layout::AB);
    set_values(input, {2, 1, -1, 4}); // [batch, features]

    const cum::Tensor output = layer.forward(input);

    REQUIRE(output.shape() == (cum::Shape{2, 2}));
    require_values(output, {0.25, 3.5, -8.75, 11});
    cum::decum();
}

TEST_CASE("Dense forward applies affine transformation and ReLU")
{
    cum::cum(cum::DEVICE::CPU);

    yann::models::layers::Dense layer(2, "relu");
    layer.init_parameters(2, 2);
    set_linear_parameters(layer);

    cum::Tensor input({2, 2}, cum::default_type, cum::layout::IO);
    set_values(input, {2, 1, -1, 4}); // [batch, features]

    const cum::Tensor output = layer.forward(input);

    REQUIRE(output.shape() == (cum::Shape{2, 2}));
    require_values(output, {0.25, 3.5, 0, 11});
    cum::decum();
}

TEST_CASE("Sequential propagates cum::Shape across layers during build")
{
    cum::cum(cum::DEVICE::CPU);

    yann::models::Sequential model({
        yann::models::layers::Input::createUnique(cum::Shape{2, 4}),
        yann::models::layers::Flatten::createUnique(),
        yann::models::layers::Dense::createUnique(16, "relu"),
        yann::models::layers::Dense::createUnique(2, "sigmoid")
    });

    REQUIRE(model.getLayersCount() == 4);

    // Layer 0: Input shape is {2, 4}, output shape is {2, 4}
    REQUIRE(model.getLayer(0)->input_shape() == (cum::Shape{2, 4}));
    REQUIRE(model.getLayer(0)->output_shape() == (cum::Shape{2, 4}));

    // Layer 1: Flatten input is {2, 4}, output is {8}
    REQUIRE(model.getLayer(1)->input_shape() == (cum::Shape{2, 4}));
    REQUIRE(model.getLayer(1)->output_shape() == (cum::Shape{8}));

    // Layer 2: Dense input is {8}, output is {16}
    REQUIRE(model.getLayer(2)->input_shape() == (cum::Shape{8}));
    REQUIRE(model.getLayer(2)->output_shape() == (cum::Shape{16}));
    REQUIRE(model.getWeights(2).rows() == 8);
    REQUIRE(model.getWeights(2).cols() == 16);

    // Layer 3: Dense input is {16}, output is {2}
    REQUIRE(model.getLayer(3)->input_shape() == (cum::Shape{16}));
    REQUIRE(model.getLayer(3)->output_shape() == (cum::Shape{2}));
    REQUIRE(model.getWeights(3).rows() == 16);
    REQUIRE(model.getWeights(3).cols() == 2);

    cum::decum();
}