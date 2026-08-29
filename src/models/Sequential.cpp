#include "Sequential.hpp"
#include "LayerType.hpp"
#include <algorithm>

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
        std::vector<Parameter*> params = this->parameters();
        bool batched = batch_size > 1;

        cum::Matrix data = X.transpose();
        cum::Matrix target = Y.transpose();

        std::vector<cum::Matrix> batches_x;
        std::vector<cum::Matrix> batches_y;
        if (batched)
        {
            for(int i = 0 ; i < X.rows() ; i += batchSize)
            {
                size_t samples = std::min(X.rows() - i, batchSize);
                cum::Matrix x_batch(X.cols(), samples, X.data() + i * X.cols());
                cum::Matrix y_batch(Y.cols(), samples, Y.data() + i * Y.cols());

                batches_x.push_back(x_batch);
                batches_y.push_back(y_batch);
            }
        }

        YANN_LOG(1, "Started training for {} epochs...", epochs);
        logging::TrainingContext ctx(*this, 0, 0, 0);
        for(size_t epoch = 0 ; epoch < epochs ; epoch++)
        {
            cum::cummulative_t totalLoss = 0;

            YANN_LOG(1, "Epoch {}", epoch);

            if(!batched)
            {
                for(int i = 0 ; i < X.rows() ; i++)
                {
                    YANN_LOG(2, "{} sample", i);

                    // cum::Matrix x = X.row(i).transpose();
                    // cum::Matrix y = Y.row(i).transpose();
                    cum::Matrix x = data.col(i);
                    cum::Matrix y = target.col(i);
                    cum::runtime::sync();

                    cum::Matrix result = this->forward(x);
                    YANN_LOG(2, "Computing loss and output gradient...", "");

                    // loss::LossType error = loss::computeLoss(result, y, this->loss_function);

                    loss.compute(result, y);
                    loss::loss_t error = loss.result();

                    YANN_LOG(2, "Performing backward pass...", "");

                    this->backward(error.gradient);
                    cum::runtime::sync();

                    YANN_LOG(2, "Updating parameters...", "");

                    optimizer.step(params);
                    YANN_LOG(2, "parameters are updated...", "");

                    totalLoss += error.value;
                }
            }
            else // batch
            {
                for(int i = 0 ; i < batches_x.size() ; i++)
                {
                    YANN_LOG(2, "{} batch", i);

                    cum::Matrix results = this->forward(batches_x[i]);

                    YANN_LOG(2, "computing loss", "");
                    loss.compute(results, batches_y[i]);

                    loss::loss_t error = loss.result();

                    YANN_LOG(2, "Performing backward pass", "");
                    this->backward(error.gradient);

                    cum::runtime::sync();

                    for(auto& callback : callbacks)
                    {
                        callback->afterBackprop(ctx);
                    }

                    YANN_LOG(2, "Updating parameters", "");
                    optimizer.step(params);

                    totalLoss += error.value;
                }
                // cum::Matrix x_batch(data.rows(), batchSize);
            }

            cum::cumeric_t avarageLoss = totalLoss / std::max(static_cast<cum::cumeric_t>(batches_x.size()), static_cast<cum::cumeric_t>(1.0));
            cum::runtime::sync();

            YANN_LOG(2, "Average epoch loss: ", static_cast<float>(avarageLoss));
            YANN_LOG(2, "Total epoch loss: ", static_cast<float>(totalLoss));

            ctx.loss = avarageLoss;
            ctx.epoch = epoch;
            ctx.batch = 1;
            for(auto& callback : callbacks)
            {
                callback->afterEpoch(ctx);
            }

            // // temporary line for debug
            // if (avarageLoss < 0.55)
            // {
            //     break;
            // }
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
        return topology[layer]->Weights();
    }

    cum::Matrix& Sequential::getBiases(size_t layer) const
    {
        return topology[layer]->Biases();
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
