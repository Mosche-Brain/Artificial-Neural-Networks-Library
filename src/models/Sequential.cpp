#include "Sequential.hpp"
#include "LayerType.hpp" 
#include "cum/Matrix.hpp"
#include <stdexcept>

#if defined(ENABLE_DEBUG_OUTPUT)
#include <iostream>
#include "utils/formating.hpp"
#endif

#include "runtime_config.hpp"
#include "cum/runtime.hpp"
#include "loss/LossBase.hpp"
// #include "utils/Logger.hpp"




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
        topology[0]->initParameters(topology[0]->size(), 1);
        for(size_t i = 1 ; i < topology.size() ; i++)
        {
            int previous_layer_size = topology[i - 1]->size();
            int current_layer_size  = topology[  i  ]->size();

            topology[i]->initParameters(current_layer_size, previous_layer_size);
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

    cum::Matrix Sequential::forward(const cum::Matrix& input)
    {
        cum::Matrix result = topology[0]->forward(input);

        for (size_t i = 1; i < topology.size(); ++i)
        {
            result = topology[i]->forward(result);
            cum::runtime::sync();
        }

        return result;
    }

    void Sequential::backward(const cum::Matrix& d_output)
    {
        if (topology.size() <= 1)
            return;

        cum::Matrix curr_gradient = topology.back()->backward(d_output);
        for (size_t i = topology.size() - 1; i > 1; --i)
        {
            cum::Matrix next_gradient = topology[i-1]->backward(curr_gradient);

            cum::runtime::sync();

            curr_gradient = std::move(next_gradient);

            cum::runtime::sync();
        }
    }

    /* TODO:
     * Przenieść budowaniu batcht do wyspecjalizowanej funkcji/klasy
     * Dodać przeładowanie pozwalające na przyjęcie zamiast X i Y zbioru batchy
     */

    void Sequential::fit(const cum::Matrix& X, const cum::Matrix& Y, loss::LossBase& loss, optimizers::OptimizerBase& optimizer, cum::dim_t epochs, cum::dim_t batch_size, std::span<logging::ITrainingCallback*> callbacks)
    {
        if (batch_size == 0)
            throw std::invalid_argument("batch_size must be greater than zero");
        if (X.cols() != Y.cols())
            throw std::invalid_argument("X and Y must contain the same number of samples");

        std::vector<Parameter*> params = this->parameters();
        const bool batched = batch_size > 1;

        std::vector<Batch> batches;
        if (batched)
        {
            for (cum::dim_t begin = 0; begin < X.cols(); begin += batch_size) // przeniósł bym tą pętle do osobnej funkcji
            {
                const cum::dim_t samples = std::min(X.cols() - begin, batch_size);
                batches.emplace_back(
                    X.slice(0, begin, X.rows(), samples),
                    Y.slice(0, begin, Y.rows(), samples),
                    Batch::ORIENTATION::COLUMN_SAMPLE);
            }
        }

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

            cum::dim_t n = batched ? batches.size() : X.cols();

            for (batch = 0; batch < n; batch++)
            {
                cum::dim_t current_batch_size = batched ? batches[batch].size : 1;
                YANN_LOG(2, "{} batch, {} samples", batch, current_batch_size);

                if (batched)
                {
                    x = batches[batch].inputs();
                    y = batches[batch].targets();
                }
                else
                {
                    x = X.col(batch);
                    y = Y.col(batch);
                }

                cum::Matrix results = this->forward(x);

                loss.compute(results, y);

                loss::loss_t error = loss.result();

                this->backward(error.gradient);

                for (logging::ITrainingCallback*&  callback : callbacks)
                    callback->afterBackprop(ctx);

                optimizer.scale_grads(params, 1 / static_cast<cum::cumeric_t>(current_batch_size));
                optimizer.step(params); // zerowanie gradientów jest dokonywanie niejawnie w kroku optymalizatora
                total_loss += error.value;
            }

            mean_loss = total_loss / static_cast<cum::cumeric_t>(batched ? batches.size() : X.cols());

            YANN_LOG(2, "Average epoch loss: ", static_cast<float>(mean_loss));
            YANN_LOG(2, "Total epoch loss: ", static_cast<float>(total_loss));


            for(logging::ITrainingCallback*& callback : callbacks)
                callback->afterEpoch(ctx);
        }
    }


    cum::Matrix& Sequential::getWeights(size_t layer) const
    {
        return topology[layer]->weights();
    }

    cum::Matrix& Sequential::getBiases(size_t layer) const
    {
        return topology[layer]->biases();
    }

    cum::Matrix& Sequential::getOutputs(size_t layer) const
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
