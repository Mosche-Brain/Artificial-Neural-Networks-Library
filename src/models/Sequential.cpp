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

    Sequential::Sequential(std::initializer_list<std::unique_ptr<layers::LayerBase>> newTopology, bool build)
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
        // cum::Matrix& curr_gradient = const_cast<cum::Matrix&>(d_output);
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


    void Sequential::fit(const cum::Matrix& X, const cum::Matrix& Y, loss::LossBase& loss, optimizers::OptimizerBase& optimizer, size_t epochs, size_t batch_size, std::span<logging::ITrainingCallback*> callbacks)
    {
        if (batch_size == 0)
            throw std::invalid_argument("batch_size must be greater than zero");

        std::vector<Parameter*> params = this->parameters();
        const bool batched = batch_size > 1;

        cum::Matrix data = X.transpose(); // zrobiłem tak bo wygodniej mi się podawało sample jako row, ale pewnie to usune
        cum::Matrix target = Y.transpose(); // w sumie niepotrzebne to

        std::vector<Batch> batches;
        if (batched)
        {
            for (std::size_t begin = 0; begin < data.cols(); begin += batch_size) // przeniósł bym tą pętle do osobnej funkcji
            {
                const std::size_t samples = std::min(data.cols() - begin, batch_size);
                batches.emplace_back(
                    data.slice(0, begin, data.rows(), samples),
                    target.slice(0, begin, target.rows(), samples),
                    Batch::ORIENTATION::COLUMN_SAMPLE);
            }
        }

        YANN_LOG(1, "Started training for {} epochs...", epochs);
        logging::TrainingContext ctx(*this, 0, 0, 0); // TODO: obecnie batch jest ignorowany, lepiej zrobię by ctx miał referencje do lossu, epoki, batcha, a nie kopie
        for(cum::dim_t epoch = 0 ; epoch < epochs ; epoch++)
        {
            cum::cummulative_t totalLoss = 0;

            YANN_LOG(1, "Epoch {}", epoch);

            // Mamy tutaj kopie, później można to na referencje zmiennić dla ograniczenia lokacji
            cum::Matrix x;
            cum::Matrix y;

            cum::dim_t n = batched ? batches.size() : X.rows();

            for (cum::dim_t i = 0; i < n; ++i)
            {
                YANN_LOG(2, "{} batch", i);

                if (batched)
                {
                    x = batches[i].inputs();
                    y = batches[i].targets();
                }
                else
                {
                    x = data.col(i);
                    y = target.col(i);
                }

                cum::Matrix results = this->forward(x);

                loss.compute(results, y);

                loss::loss_t error = loss.result();

                this->backward(error.gradient); // loss sam skaluje gradient

                if (batched)
                    for (Parameter* param : params) { param->scale_gradient(static_cast<cum::cumeric_t>(1) / static_cast<cum::cumeric_t>(batches[i].size));

                for (auto& callback : callbacks)
                    callback->afterBackprop(ctx);

                optimizer.step(params); // zerowanie gradientów jest dokonywanie niejawnie w kroku optymalizatora
                totalLoss += error.value;
            }

            cum::cumeric_t avarageLoss = totalLoss / static_cast<cum::cumeric_t>(batched ? batches.size() : X.rows()));

            YANN_LOG(2, "Average epoch loss: ", static_cast<float>(avarageLoss));
            YANN_LOG(2, "Total epoch loss: ", static_cast<float>(totalLoss));

            ctx.loss = avarageLoss;
            ctx.epoch = epoch;
            // ctx.batch = 1;
            for(auto& callback : callbacks)
            {
                callback->afterEpoch(ctx);
            }
        }
    }

    //void Sequential::updateParams(cum::cumeric_t rate) // depraced
    //{
    //    for(size_t i = 0 ; i < topology.size() ; i++)
    //    {
    //        if(topology[i]->layerType() != layers::LAYER_TYPE::INPUT)
    //            topology[i]->update_weights(rate);
    //    }
    //}

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
