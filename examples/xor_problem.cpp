#include <cum/Matrix.hpp>
#include <cum/cum.hpp>
#include <cum/runtime.hpp>

#include <yann/logging/ITrainingCallback.hpp>
#include <yann/logging/LossTracker.hpp>
#include <yann/loss/BinaryCrossEntropy.hpp>
#include <yann/models/Sequential.hpp>
#include <yann/optimizers/SGD.hpp>
#include <yann/runtime_config.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <algorithm>
#include <array>
class GradientEpochTracker final : public yann::logging::ITrainingCallback
{
public:
    GradientEpochTracker()
    {
        std::ofstream("plots/yann_gradient_trace.txt", std::ios::trunc);
    }

    void beforeEpoch(yann::logging::TrainingContext&) override
    {
    }

    void afterBackprop(yann::logging::TrainingContext& context) override
    {
        const auto& topology = context.model.getTopology();
        if (weights_sum_.size() != topology.size())
        {
            weights_sum_.assign(topology.size(), 0);
            biases_sum_.assign(topology.size(), 0);
        }

        for (std::size_t layer = 1; layer < topology.size(); ++layer)
        {
            weights_sum_[layer] += topology[layer]->WeightsGrad().amean();
            biases_sum_[layer] += topology[layer]->BiasesGrad().amean();
        }
        ++backprop_count_;
    }

    void afterEpoch(yann::logging::TrainingContext& context) override
    {
        std::ofstream file("plots/yann_gradient_trace.txt", std::ios::app);
        const auto& topology = context.model.getTopology();
        for (std::size_t layer = 1; layer < topology.size(); ++layer)
        {
            file << context.epoch << ' ' << layer << ' '
                 << static_cast<double>(weights_sum_[layer] / backprop_count_) << ' '
                 << static_cast<double>(biases_sum_[layer] / backprop_count_) << '\n';
        }
        std::fill(weights_sum_.begin(), weights_sum_.end(), 0);
        std::fill(biases_sum_.begin(), biases_sum_.end(), 0);
        backprop_count_ = 0;
    }

private:
    std::vector<cum::cumeric_t> weights_sum_;
    std::vector<cum::cumeric_t> biases_sum_;
    std::size_t backprop_count_ = 0;
};


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

    // Same deterministic start as examples/xor_gradient_reference.py.
    set_values(model.getWeights(1), {
        -0.37002477_c, 0.99407870_c,
        0.41585749_c, -0.61816370_c});
    set_values(model.getBiases(1), {0.0_c, 0.0_c});
    set_values(model.getWeights(2), {0.67214078_c, -1.45004952_c});
    set_values(model.getBiases(2), {0.0_c});

    const cum::Matrix predictions_before = model.forward(inputs.transpose());
    print_predictions("before training:", inputs, predictions_before);

    yann::loss::Loss loss = yann::loss::BinaryCrossEntropy::create();
    yann::optimizers::Optimizer optimizer = yann::optimizers::SGD::create(1.0_c);
    yann::logging::LossTracker loss_tracker;
    GradientEpochTracker gradient_tracker;
    std::array<yann::logging::ITrainingCallback*, 2> callbacks = {
        &loss_tracker,
        &gradient_tracker};

    constexpr std::size_t epochs = 100;
    model.fit(inputs, targets, *loss, *optimizer, epochs, 1, callbacks);
    cum::runtime::sync();

    const cum::Matrix predictions_after = model.forward(inputs.transpose());
    print_predictions("after training:", inputs, predictions_after);

    std::cout << "loss: "
              << loss_tracker.getLossHistory().front()
              << " -> "
              << loss_tracker.getLossHistory().back()
              << '\n';

    cum::decum();
    return 0;
}
