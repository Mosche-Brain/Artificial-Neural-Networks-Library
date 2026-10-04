#include <stdexcept>
#include <ranges>
#include <print>

#include <cum/runtime.hpp>

#if defined(ENABLE_DEBUG_OUTPUT)
#include <iostream>
#include "utils/formating.hpp"
#endif

#include "yann/models/layers/LayerType.hpp"

#include "yann/loss/LossBase.hpp"
#include "yann/runtime_config.hpp"

#include "yann/models/Sequential.hpp"

/* TODO:
 *  Dobrze było by ustalić sposób na niesienie informacji o tym, która oś tensora jest batchem, można by to zrobić jako globalną zmienną w singletonowym runtime_config, chociaż to było by dziwne.
 */

namespace yann::models
{
    Sequential::Sequential()
    {

    }

    Sequential::Sequential(std::initializer_list<std::unique_ptr<layers::LayerBase>> newTopology, bool build) // This constructor is awesome, I reject every other opinion
    {
        YANN_LOG(1, "Initializing Sequential model with {} layers...", newTopology.size());

        topology.reserve(newTopology.size());
        for(auto& ptr : newTopology)
        {
            topology.push_back(std::move(const_cast<std::unique_ptr<layers::LayerBase>&>(ptr))); // This work properly with fresh layers pointers created by ::createUnique(...) fabriques
        }

        if(!build) return;

        this->build();
    }

    void Sequential::build()
    {
        topology[0]->init_parameters(topology[0]->size(), 1);

        for(auto [index, layer] : topology | std::views::enumerate)
        {
            if (index == 0) continue;

            int previous_layer_size = topology[index - 1]->size();

            layer->init_parameters(layer->size(), previous_layer_size);
        }
    }

    void Sequential::addLayer(LayerPtr layer)
    {
        topology.push_back(std::move(layer));
    }

    void Sequential::clear()
    {
        topology.clear();
    }

    cum::Tensor Sequential::forward(const cum::Tensor& input)
    {
        cum::Tensor result = input;
        // std::println("ok");
        for (auto [index, layer] : topology | std::views::enumerate)
        {
            // std::println("forward layerd {}", index);
            result = layer->forward(result);
            // result = layer->cache.a;
        }

        return result;
    }

    void Sequential::backward(const cum::Tensor& d_output)
    {
        if (topology.empty()) return;


        cum::Tensor curr_gradient = d_output;
        for (auto [index, layer] : topology | std::views::enumerate | std::views::reverse)
        {
            if (layer->layerType() == layers::LayerType::Input)
                break;

            cum::Tensor next_gradient = layer->backward(curr_gradient);

            curr_gradient = std::move(next_gradient);
            // YANN_LOG(2, "||dL/dX||: {},\t mean|dL/dX|: {}", curr_gradient.norm(), curr_gradient.abs().mean());
        }
    }

    /* TODO: (obsolete)
     *  budowaniu batchy do wyspecjalizowanej funkcji/klasy
     *  Dodać przeładowanie pozwalające na przyjęcie zamiast X i Y zbioru batchy
     */

    void Sequential::fit(const cum::Tensor& X, const cum::Tensor& Y, loss::LossBase& loss, optimizers::OptimizerBase& optimizer, cum::dim_t epochs, cum::dim_t batch_size, std::span<logging::ITrainingCallback*> callbacks)
    {
        constexpr cum::dim_t batch_axis = 1; // tymczasowo fixed

        if (batch_size == 0)
            throw std::invalid_argument("batch_size must be greater than zero");
        if (X.shape()[1] != Y.shape()[1])
            throw std::invalid_argument("X and Y must contain the same number of samples");

        std::vector<Parameter*> params = this->parameters();
        const bool batched = batch_size > 1;

        YANN_LOG(1, "Started training for {} epochs...", epochs);

        cum::cumeric_t mean_loss_prev = 0;
        cum::cumeric_t mean_loss = 0;
        cum::dim_t epoch = 0;
        cum::dim_t batch = 0;

        logging::TrainingContext ctx(*this, mean_loss, epoch, batch);
        for( ; epoch < epochs ; epoch++)
        {
            cum::cummulative_t total_loss = 0;


            cum::Shape input_shape = topology.front()->input_shape();
            cum::Shape output_shape = topology.back()->output_shape();


            input_shape.push_back(batch_size);
            output_shape.push_back(batch_size);


            cum::Tensor x(input_shape, cum::default_type);
            cum::Tensor y(output_shape, cum::default_type);

            cum::dim_t n = X.cols(); // assuming batch last logic layout, and 2D tensors
            cum::dim_t batches = n / batch_size;
            n % batch_size > 0 ? batches++ : NULL;
            for (batch = 0; batch < batches; batch++)
            {
                cum::dim_t num_samples = (batch + 1) * batch_size > n ? n % batch_size : batch_size;
                cum::dim_t sample_idx = batch * batch_size;
                YANN_LOG(3, "batch: {}", batch);

                cum::Shape input_batch_shape = input_shape;
                input_batch_shape.at(input_batch_shape.size() - 1) = num_samples;

                cum::Shape input_batch_indices(input_shape.size(), 0);
                input_batch_indices[batch_axis] = sample_idx;

                x = X.slice(input_batch_indices, input_batch_shape);
                YANN_LOG(3, "X: Sample: {}x{}", x.rows(), x.cols());

                cum::Shape target_batch_shape = output_shape;
                target_batch_shape.at(target_batch_shape.size() - 1) = num_samples;

                cum::Shape target_batch_indices(target_batch_shape.size(), 0);
                target_batch_indices.at(batch_axis) = sample_idx;

                y = Y.slice(target_batch_indices, target_batch_shape);
                YANN_LOG(3, "Y: Sample: {}x{}", y.rows(), y.cols());

                cum::runtime::sync();
                cum::Tensor results = this->forward(x);
                YANN_LOG(2, "results: {}x{} | norm: {} | mean: {}", results.rows(), results.cols(), results.norm(), results.mean());


                loss.compute(results, y);

                auto [error, gradient] = loss.result();

                YANN_LOG(2, "error gradient | norm: {} | mean: {}", error, gradient.norm(), gradient.mean());

                this->backward(gradient);

                for (logging::ITrainingCallback*&  callback : callbacks)
                    callback->afterBackprop(ctx);

                optimizer.step(params); // zerowanie gradientów jest dokonywanie niejawnie w kroku optymalizatora
                total_loss += error;
            }

            mean_loss_prev = mean_loss;
            mean_loss = total_loss / static_cast<cum::cumeric_t>(batches);

            cum::cumeric_t diff_loss = mean_loss - mean_loss_prev;
            YANN_LOG(1, "Epoch: {}\t| Loss: {}\t| dL: {}", epoch, mean_loss, diff_loss);


            for(logging::ITrainingCallback*& callback : callbacks)
                callback->afterEpoch(ctx);
        }
    }


    cum::Tensor& Sequential::getWeights(size_t layer) const
    {
        return topology[layer]->weights();
    }

    cum::Tensor& Sequential::getBiases(size_t layer) const
    {
        return topology[layer]->biases();
    }

    cum::Tensor& Sequential::getOutputs(size_t layer) const
    {
        return topology[layer]->Outputs();
    }

    cum::functions::activation_t Sequential::getActivation(size_t layer) const
    {
        return topology[layer]->activation;
    }

    auto Sequential::getLayer(size_t layer) const -> LayerPtr // currently not used
    {
        // return std::move(topology[layer]);
    }

    auto Sequential::getTopology() const -> const Topology&
    {
        return topology;
    }

    size_t Sequential::getLayersCount() const
    {
        return topology.size();
    }

    int Sequential::getLayerSize(size_t layer) const
    {
        return topology[layer]->size();
    }

    std::vector<Parameter*> Sequential::parameters()
    {
        std::vector<Parameter*> params;

        for(auto& layer : topology)
            layer->collect_parameters(params);

        return params;
    }
}
