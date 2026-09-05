#include <cum/Matrix.hpp>
#include <cum/cum.hpp>
#include <cum/runtime.hpp>

#include <yann/logging/ITrainingCallback.hpp>
#include <yann/logging/LossTracker.hpp>
#include <yann/loss/BinaryCrossEntropy.hpp>
#include <yann/models/Sequential.hpp>
#include <yann/optimizers/SGD.hpp>
#include <yann/runtime_config.hpp>

#include <array>
#include <matplot/matplot.h>

#include <numeric>
#include <chrono>
#include <vector>


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

    // The source literals use rows; transpose them into column-sample layout.
    cum::Matrix inputs(4, 2, {
        0.0_c, 0.0_c,
        0.0_c, 1.0_c,
        1.0_c, 0.0_c,
        1.0_c, 1.0_c
    });
    cum::Matrix targets(4, 1, {
        0.0_c,
        1.0_c,
        1.0_c,
        0.0_c
    });

    inputs.transposeInPlace();
    targets.transposeInPlace();


    // return 0;
    // Same deterministic start as examples/xor_gradient_reference.py.
    set_values(model.getWeights(1), {
        -0.37002477_c, 0.99407870_c,
        0.41585749_c, -0.61816370_c});
    set_values(model.getBiases(1), {0.0_c, 0.0_c});
    set_values(model.getWeights(2), {0.67214078_c, -1.45004952_c});
    set_values(model.getBiases(2), {0.0_c});

    const cum::Matrix predictions_before = model.forward(inputs);
    print_predictions("before training:", inputs.transpose(), predictions_before);

    yann::loss::Loss loss = yann::loss::BinaryCrossEntropy::create();
    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(0.1_c);
    yann::logging::LossTracker loss_tracker;
    std::array<yann::logging::ITrainingCallback*, 1> callbacks = {&loss_tracker};

    constexpr std::size_t epochs = 100;
    cum::runtime::sync();
    const auto training_start = std::chrono::steady_clock::now();
    model.fit(inputs, targets, *loss, *optimizer, epochs, 1, callbacks);
    cum::runtime::sync();
    const auto training_end = std::chrono::steady_clock::now();

    const std::chrono::duration<double> training_time =
        training_end - training_start;
    std::cout << "training time: "
              << training_time.count()
              << " s\n";

    const cum::Matrix predictions_after = model.forward(inputs);
    print_predictions("after training:", inputs.transpose(), predictions_after);

    std::cout << "loss: "
              << loss_tracker.getLossHistory().front()
              << " -> "
              << loss_tracker.getLossHistory().back()
              << '\n';

    const auto& loss_history = loss_tracker.getLossHistory();
    std::vector<double> epoch_range(loss_history.size());
    std::iota(epoch_range.begin(), epoch_range.end(), 0.0);

    auto figure = matplot::figure(true);
    matplot::plot(epoch_range, loss_history);
    matplot::title("XOR training loss");
    matplot::xlabel("epoch");
    matplot::ylabel("loss");
    figure->size(1200, 800);
    matplot::show();




              
    cum::decum();
    return 0;
}
