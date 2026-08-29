#include <cum/Matrix.hpp>
#include <cum/cum.hpp>
#include <cum/runtime.hpp>

#include <yann/logging/LossTracker.hpp>
#include <yann/loss/BinaryCrossEntropy.hpp>
#include <yann/models/Sequential.hpp>
#include <yann/optimizers/SGD.hpp>
#include <yann/runtime_config.hpp>

#include <iomanip>
#include <iostream>
#include <string_view>

namespace
{
    void set_values(cum::Matrix& matrix, std::initializer_list<cum::cumeric_t> values)
    {
        std::size_t index = 0;
        for (const auto value : values)
        {
            matrix.data()[index++] = value;
        }
    }

    void print_predictions(std::string_view title,
                           const cum::Matrix& inputs,
                           const cum::Matrix& predictions)
    {
        cum::runtime::sync();
        std::cout << title << '\n';
        std::cout << std::setprecision(6);
        for (std::size_t sample = 0; sample < inputs.rows(); ++sample)
        {
            std::cout << '[' << inputs(sample, 0) << ',' << inputs(sample, 1)
                      << "] -> " << predictions(0, sample) << '\n';
        }
    }
}

int main()
{
    cum::cum(cum::CUM_DEVICE::CPU);
    yann::runtime_config::set_verbosity(0);

    yann::models::Sequential model;
    model.addLayer(yann::models::layers::Input::createUnique(2));
    model.addLayer(yann::models::layers::Dense::createUnique(2, "tanh"));
    model.addLayer(yann::models::layers::Dense::createUnique(1, "sigmoid"));
    model.build();

    // Four samples are stored as rows for fit(); forward receives their transpose.
    const cum::Matrix inputs(4, 2, {
        0.0_c, 0.0_c,
        0.0_c, 1.0_c,
        1.0_c, 0.0_c,
        1.0_c, 1.0_c
    });
    const cum::Matrix targets(4, 1, {
        0.0_c,
        1.0_c,
        1.0_c,
        0.0_c
    });

    // Fixed parameters make the example reproducible.
    set_values(model.getWeights(1), {4.0_c, 4.0_c, -4.0_c, -4.0_c});
    set_values(model.getBiases(1), {-2.0_c, 6.0_c});
    set_values(model.getWeights(2), {5.0_c, 5.0_c});
    set_values(model.getBiases(2), {-2.5_c});

    const cum::Matrix predictions_before = model.forward(inputs.transpose());
    print_predictions("before training:", inputs, predictions_before);

    yann::loss::Loss loss = yann::loss::BinaryCrossEntropy::create();
    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(0.1_c);
    yann::logging::LossTracker loss_tracker;
    std::array<yann::logging::ITrainingCallback*, 1> callbacks = {&loss_tracker};

    model.fit(inputs, targets, *loss, *optimizer, 1, callbacks);
    cum::runtime::sync();

    const cum::Matrix predictions_after = model.forward(inputs.transpose());
    print_predictions("after one training epoch:", inputs, predictions_after);

    std::cout << "loss: " << loss_tracker.getLossHistory().back() << '\n';

    cum::decum();
    return 0;
}
