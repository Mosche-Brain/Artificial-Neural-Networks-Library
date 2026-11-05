#include <ranges>
#include <stdexcept>

#include <cum/runtime.hpp>

#include "yann/models/Chain.hpp"
#include "yann/runtime_config.hpp"

namespace yann::models
{
    Chain::Chain(std::initializer_list<ModelRef> models)
        : models_(models)
    {
    }

    Chain& Chain::addModel(Sequential& model)
    {
        models_.emplace_back(model);
        return *this;
    }

    void Chain::build(const cum::Shape& input_shape)
    {
        if (models_.empty())
            return;

        cum::Shape current_shape = input_shape;
        for (Sequential& model : models_)
        {
            model.build(current_shape);
            if (model.getLayersCount() > 0)
            {
                current_shape = model.getTopology().back()->output_shape();
            }
        }
    }

    cum::Tensor Chain::forward(const cum::Tensor& input)
    {
        cum::Tensor result = input;
        for (Sequential& model : models_)
        {
            result = model.forward(result);
        }
        return result;
    }

    cum::Tensor Chain::backward(const cum::Tensor& d_output)
    {
        if (models_.empty())
            return d_output;

        cum::Tensor curr_gradient = d_output;
        for (Sequential& model : models_ | std::views::reverse)
        {
            curr_gradient = model.backward(curr_gradient);
        }
        return curr_gradient;
    }

    std::vector<Parameter*> Chain::parameters() const
    {
        std::vector<Parameter*> params;
        for (Sequential& model : models_)
        {
            auto model_params = model.parameters();
            params.insert(params.end(), model_params.begin(), model_params.end());
        }
        return params;
    }

    Sequential& Chain::operator[](const std::size_t index) const
    {
        if (index >= models_.size())
            throw std::out_of_range("Chain::operator[]: model index out of range");
        return models_[index].get();
    }

    void Chain::fit(
        const cum::Tensor& X,
        const cum::Tensor& Y,
        loss::LossBase& loss,
        optimizers::OptimizerBase& optimizer,
        const cum::dim_t epochs,
        const cum::dim_t batch_size,
        std::span<logging::ITrainingCallback*> callbacks
    )
    {
        constexpr cum::dim_t batch_axis = 0;

        if (models_.empty())
            throw std::logic_error("Chain::fit: cannot train an empty Chain");
        if (batch_size == 0)
            throw std::invalid_argument("batch_size must be greater than zero");
        if (X.shape()[batch_axis] != Y.shape()[batch_axis])
            throw std::invalid_argument("X and Y must contain the same number of samples");

        Sequential& first_model = models_.front().get();
        Sequential& last_model  = models_.back().get();
        if (first_model.getLayersCount() == 0 || last_model.getLayersCount() == 0)
            throw std::logic_error("Chain::fit: chained models must not be empty");

        std::vector<Parameter*> params = this->parameters();
        YANN_LOG(1, "Chain: Found {} parameter tensors across {} models", params.size(), models_.size());

        cum::cumeric_t mean_loss_prev = 0;
        cum::cumeric_t mean_loss = 0;
        cum::dim_t epoch = 0;
        cum::dim_t batch = 0;

        logging::TrainingContext ctx(last_model, mean_loss, epoch, batch);

        cum::Shape input_shape  = first_model.getTopology().front()->input_shape();
        cum::Shape output_shape = last_model.getTopology().back()->output_shape();

        input_shape.insert(input_shape.begin(), batch_size);
        output_shape.insert(output_shape.begin(), batch_size);

        const cum::dim_t n = X.shape()[batch_axis];
        cum::dim_t batches = n / batch_size;
        if (n % batch_size > 0)
            ++batches;

        for (; epoch < epochs; ++epoch)
        {
            for (logging::ITrainingCallback*& callback : callbacks)
                callback->beforeEpoch(ctx);

            cum::cummulative_t total_loss = 0;

            for (batch = 0; batch < batches; ++batch)
            {
                const cum::dim_t num_samples = (batch + 1) * batch_size > n ? n % batch_size : batch_size;
                const cum::dim_t sample_idx  = batch * batch_size;

                cum::Shape input_batch_shape = input_shape;
                input_batch_shape.at(batch_axis) = num_samples;
                cum::Shape input_batch_indices(input_shape.size(), 0);
                input_batch_indices[batch_axis] = sample_idx;

                cum::Tensor x = X.slice(input_batch_indices, input_batch_shape);

                cum::Shape target_batch_shape = output_shape;
                target_batch_shape.at(batch_axis) = num_samples;
                cum::Shape target_batch_indices(target_batch_shape.size(), 0);
                target_batch_indices.at(batch_axis) = sample_idx;

                cum::Tensor y = Y.slice(target_batch_indices, target_batch_shape);

                cum::runtime::sync();
                cum::Tensor results = this->forward(x);

                loss.compute(results, y);
                const auto& [error, gradient] = loss.result();

                this->backward(gradient);

                for (logging::ITrainingCallback*& callback : callbacks)
                    callback->afterBackprop(ctx);

                optimizer.step(params);
                total_loss += error;
            }

            mean_loss_prev = mean_loss;
            mean_loss = total_loss / static_cast<cum::cumeric_t>(batches);

            const cum::cumeric_t diff_loss = mean_loss - mean_loss_prev;
            YANN_LOG(1, "Epoch: {}\t| Loss: {}\t| dL: {}", epoch, mean_loss, diff_loss);

            for (logging::ITrainingCallback*& callback : callbacks)
                callback->afterEpoch(ctx);
        }
    }
} // namespace yann::models
