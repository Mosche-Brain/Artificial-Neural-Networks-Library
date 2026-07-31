#include "Sequential.hpp"
#include "LayerType.hpp"

#include <algorithm>
#include <random>

#if defined(ENABLE_DEBUG_OUTPUT)
    #include <iostream>
    #include "utils/formating.hpp"
#endif

#include "runtime_config.hpp"
#include "cum/runtime.hpp"
// #include "utils/Logger.hpp"

#define DEFAULT_LOSS_FUNC loss::LossFunction::mse

namespace yann::models
{
    Sequential::Sequential()
    {

    }

    Sequential::Sequential(std::initializer_list<std::unique_ptr<layers::LayerBase>> newTopology) : loss_function(DEFAULT_LOSS_FUNC)
    {
        YANN_LOG(1, "Initializing Sequential model with {} layers...", newTopology.size());

        topology.reserve(newTopology.size()); 
        for(auto& ptr : newTopology) 
        {
            topology.push_back(std::move(const_cast<std::unique_ptr<layers::LayerBase>&>(ptr)));
        }
        
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

        if(topology.size() == 1)
        {
            size_t layerSize = topology.back()->size();
            topology.back()->initParameters(layerSize, 1);
        }
        else
        {
            size_t layerSize  = topology.back()->size();
            size_t inputWidth = topology[topology.size() - 1]->size();
            topology.back()->initParameters(layerSize, inputWidth );
        }
    }

    void Sequential::clear()
    {
        topology.clear();
    }

    void Sequential::setLossFunction(loss::LossFunction new_loss_function)
    {
        loss_function = new_loss_function;
    }

    cum::Matrix Sequential::forward(const cum::Matrix& input)
    {
        if (input.cols() == 1) // single sample
        {
            topology[0]->forward(input);

            for(size_t i = 1 ; i < topology.size() ; ++i)
            {
                topology[i]->forward(topology[i - 1]->Outputs());
            }

            return topology.back()->Outputs();
        }
        else // batch
        {
            cum::Matrix result = input;
            for(size_t i = 0 ; i < topology.size() ; ++i)
            {
                result = topology[i]->forward(result);
            }
            return result;
        }
    }
    void Sequential::backward(const cum::Matrix& d_output)
    {
        cum::Matrix& curr_gradient = const_cast<cum::Matrix&>(d_output);

        for(size_t i = topology.size() - 1 ; i > 0 ; --i)
        {
            // if(topology[i]->layerType() == layers::LAYER_TYPE::INPUT)
            //     continue;

            curr_gradient = topology[i]->backward(curr_gradient);
        }
        // if(runtime_config::verbosity_level() >= 3)
        // {
        //     std::cout << "\t\t\t" << "layer 0 gradient: " << utils::formating::matrixToString(curr_gradient.transpose()) << '\n';
        // }
    }


    void Sequential::fit(const cum::Matrix& X, const cum::Matrix& Y, optimizers::OptimizerBase& optimizer, size_t epochs, std::span<logging::ITrainingCallback*> callbacks)
    {
        std::vector<Parameter*> params = this->parameters();

        YANN_LOG(1, "Started training for {} epochs...", epochs);
        for(size_t epoch = 0 ; epoch < epochs ; epoch++)
        {
            cum::cummulative_t totalLoss = 0;

            YANN_LOG(1, "Epoch {}", epoch);

            for(int i = 0 ; i < X.rows() ; i++)
            {
                YANN_LOG(2, "{} sample", i);

                cum::Matrix x = X.row(i).transpose();
                cum::Matrix y = Y.row(i).transpose();
                cum::runtime::sync();

                cum::Matrix result = this->forward(x);
                YANN_LOG(2, "Computing loss and output gradient...", "");

                loss::LossType error = loss::computeLoss(result, y, this->loss_function);

                YANN_LOG(2, "Performing backward pass...", "");

                this->backward(error.gradient);
                cum::runtime::sync();

                YANN_LOG(2, "Updating parameters...", "");

                optimizer.step(params);
                YANN_LOG(2, "parameters are updated...", "");

                totalLoss += error.loss;
            }
            cum::cumeric_t avarageLoss = totalLoss / X.rows();
            cum::runtime::sync();


            YANN_LOG(2, "Average epoch loss: ", static_cast<float>(avarageLoss));
            YANN_LOG(2, "Total epoch loss: ", static_cast<float>(totalLoss));

            logging::TrainingContext ctx(*this, avarageLoss, epoch, 1);
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

    void Sequential::updateParams(cum::cumeric_t rate) // depraced
    {
        for(size_t i = 0 ; i < topology.size() ; i++)
        {
            if(topology[i]->layerType() != layers::LAYER_TYPE::INPUT)
                topology[i]->update_weights(rate);
        }
    }

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

    auto Sequential::getTopology() const -> Topology // currently not used
    {
        // Topology temp;
        // for(size_t i = 0 ; i < topology.size() ; i++)
        // {
            // temp.push_back(std::move(topology[i]));
        // }
        // return temp;
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