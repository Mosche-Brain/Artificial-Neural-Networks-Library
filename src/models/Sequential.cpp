#include "LayerType.hpp"
#include <cum/Matrix.hpp>
#include <stdexcept>
#include <ranges>

#include <cum/runtime.hpp>

#if defined(ENABLE_DEBUG_OUTPUT)
#include <iostream>
#include "utils/formating.hpp"
#endif

#include "yann/loss/LossBase.hpp"
#include "yann/runtime_config.hpp"

#include "yann/models/Sequential.hpp"

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
            topology.push_back(std::move(const_cast<std::unique_ptr<layers::LayerBase>&>(ptr)));
        }

        if(!build) return;

        this->build();
    }

    void Sequential::build()
    {
        topology[0]->init_parameters(topology[0]->size(), 1);
        // for(size_t i = 1 ; i < topology.size() ; i++)
        for(auto [index, layer] : topology | std::views::enumerate)
        {
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
        cum::Tensor result = topology[0]->forward(input);
        for (auto& layer : topology)
        {
            result = layer->forward(result);
        }

        return result;
    }

    void Sequential::backward(const cum::Tensor& d_output)
    {
        if (topology.empty()) return;

        cum::Tensor curr_gradient = topology.back()->backward(d_output);
        // for (size_t i = topology.size() - 1; i > 0; --i)
        for (auto [index, layer] : topology | std::views::enumerate | std::views::reverse)
        {
            if (layer->layerType() == layers::LayerType::Input)
                break;

            cum::Tensor next_gradient = layer->backward(curr_gradient);


            curr_gradient = std::move(next_gradient);

        }
    }

    /* TODO: (obsolete)
     * Przenieść budowaniu batcht do wyspecjalizowanej funkcji/klasy
     * Dodać przeładowanie pozwalające na przyjęcie zamiast X i Y zbioru batchy
     */

    void Sequential::fit(const cum::Tensor& X, const cum::Tensor& Y, loss::LossBase& loss, optimizers::OptimizerBase& optimizer, cum::dim_t epochs, cum::dim_t batch_size, std::span<logging::ITrainingCallback*> callbacks)
    {
        if (batch_size == 0)
            throw std::invalid_argument("batch_size must be greater than zero");
        // if (X.cols() != Y.cols())
        if (X.shape()[1] != Y.shape()[1])
            throw std::invalid_argument("X and Y must contain the same number of samples");

        std::vector<Parameter*> params = this->parameters();
        const bool batched = batch_size > 1;

        // std::vector<Batch> batches;
        // if (batched)
        // {
        //     for (cum::dim_t begin = 0; begin < X.cols(); begin += batch_size) // przeniósł bym tą pętle do osobnej funkcji
        //     {
        //         const cum::dim_t samples = std::min(X.cols() - begin, batch_size);
        //         batches.emplace_back(
        //             X.slice(0, begin, X.rows(), samples),
        //             Y.slice(0, begin, Y.rows(), samples),
        //             Batch::ORIENTATION::COLUMN_SAMPLE);
        //     }
        // }

        YANN_LOG(1, "Started training for {} epochs...", epochs);

        cum::cumeric_t mean_loss = 0;
        cum::dim_t epoch = 0;
        cum::dim_t batch = 0;


        logging::TrainingContext ctx(*this, mean_loss, epoch, batch); // TODO: obecnie batch jest ignorowany, lepiej zrobię by ctx miał referencje do lossu, epoki, batcha, a nie kopie
        for(epoch = 0 ; epoch < epochs ; epoch++)
        {
            cum::cummulative_t total_loss = 0;

            YANN_LOG(1, "Epoch {}", epoch);

            // Mamy tutaj kopie, później można to na referencje zmiennić dla ograniczenia lokacji
            cum::Matrix x;
            cum::Matrix y;

            // cum::dim_t n = batched ? batches.size() : X.cols();
            cum::dim_t n = X.shape()[1];
            for (batch = 0; batch < n; batch++)
            {
                // cum::dim_t current_batch_size = batched ? batches[batch].size : 1;
                // YANN_LOG(2, "{} batch, {} samples", batch, current_batch_size);

                if (batched)
                {
                    // x = batches[batch].inputs();
                    // y = batches[batch].targets();
                }
                else
                {
                    // x = X.col(batch);
                    // y = Y.col(batch);
                }

                // cum::Matrix results = this->forward(x);

                // loss.compute(results, y);

                // loss::loss_t error = loss.result();

                // this->backward(error.gradient);

                for (logging::ITrainingCallback*&  callback : callbacks)
                    callback->afterBackprop(ctx);

                // optimizer.scale_grads(params, 1 / static_cast<cum::cumeric_t>(current_batch_size));
                optimizer.step(params); // zerowanie gradientów jest dokonywanie niejawnie w kroku optymalizatora
                // total_loss += error.value;
            }

            // mean_loss = total_loss / static_cast<cum::cumeric_t>(batched ? batches.size() : X.cols());

            YANN_LOG(2, "Average epoch loss: ", static_cast<float>(mean_loss));
            YANN_LOG(2, "Total epoch loss: ", static_cast<float>(total_loss));


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
