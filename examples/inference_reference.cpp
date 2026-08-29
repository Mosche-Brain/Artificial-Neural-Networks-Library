#include <cum/cum.hpp>
#include <cum/Matrix.hpp>
#include <cum/runtime.hpp>

#include <yann/models/Sequential.hpp>

#include <iomanip>
#include <iostream>
#include <string_view>

namespace
{
    void print_matrix(std::string_view name, const cum::Matrix& matrix)
    {
        cum::runtime::sync();
        std::cout << name << ' ' << matrix.rows() << 'x' << matrix.cols() << '\n';
        std::cout << std::setprecision(9);
        for (std::size_t row = 0; row < matrix.rows(); ++row)
        {
            for (std::size_t col = 0; col < matrix.cols(); ++col)
            {
                std::cout << static_cast<double>(matrix(row, col));
                if (col + 1 != matrix.cols())
                {
                    std::cout << ' ';
                }
            }
            std::cout << '\n';
        }
    }

    void set_matrix(cum::Matrix& matrix, std::initializer_list<cum::cumeric_t> values)
    {
        std::size_t index = 0;
        for (const auto value : values)
        {
            matrix.data()[index++] = value;
        }
    }

    yann::models::Sequential make_single_layer_model()
    {
        yann::models::Sequential model;
        model.addLayer(yann::models::layers::Input::createUnique(2));
        model.addLayer(yann::models::layers::Dense::createUnique(2, "linear"));
        model.build();

        set_matrix(model.getWeights(1), {1.0_c, 2.0_c, -1.0_c, 0.5_c});
        set_matrix(model.getBiases(1), {0.5_c, -2.0_c});
        return model;
    }

    void single_sample()
    {
        auto model = make_single_layer_model();
        cum::Matrix input(2, 1, {2.0_c, -1.0_c});
        print_matrix("single_sample_input", input);
        print_matrix("single_sample_output", model.forward(input));
    }

    void batch()
    {
        auto model = make_single_layer_model();
        cum::Matrix input(2, 3, {1.0_c, 2.0_c, 3.0_c,
                                 4.0_c, 5.0_c, 6.0_c});
        print_matrix("batch_input", input);
        print_matrix("batch_output", model.forward(input));
    }

    void two_layers()
    {
        yann::models::Sequential model;
        model.addLayer(yann::models::layers::Input::createUnique(2));
        model.addLayer(yann::models::layers::Dense::createUnique(2, "relu"));
        model.addLayer(yann::models::layers::Dense::createUnique(1, "linear"));
        model.build();

        set_matrix(model.getWeights(1), {1.0_c, -1.0_c, 0.5_c, 2.0_c});
        set_matrix(model.getBiases(1), {0.0_c, 1.0_c});
        set_matrix(model.getWeights(2), {2.0_c, -3.0_c});
        set_matrix(model.getBiases(2), {0.25_c});

        cum::Matrix input(2, 2, {2.0_c, -1.0_c,
                                 1.0_c, 3.0_c});
        print_matrix("two_layers_input", input);
        print_matrix("two_layers_output", model.forward(input));
    }
}

int main()
{
    cum::cum(cum::CUM_DEVICE::CPU);
    single_sample();
    batch();
    two_layers();
    cum::decum();
    return 0;
}
