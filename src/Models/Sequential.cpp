#include "Sequential.hpp"
#include "LayerType.hpp"

#include <algorithm>
#include <random>

#if defined(ENABLE_DEBUG_OUTPUT)
    #include <iostream>
    #include "Utility/logs.hpp"
#endif

#include "runtime_config.hpp"

#define DEFAULT_LOSS_FUNC utils::loss::LossFunction::mse

namespace yann::models
{
    Sequential::Sequential()
    {

    }

    Sequential::Sequential(std::initializer_list<std::unique_ptr<layers::LayerBase>> newTopology) : loss_function(DEFAULT_LOSS_FUNC)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 1)
            std::cout << "Initializing Sequential model with " << newTopology.size() << " layers...\n";
        #endif

        topology.reserve(newTopology.size()); 
        for(auto& ptr : newTopology) 
        {
            topology.push_back(std::move(const_cast<std::unique_ptr<layers::LayerBase>&>(ptr)));
        }
        
        topology[0]->initParameters(topology[0]->size(), 1);
        for(size_t i = 1 ; i < topology.size() ; i++)
        {
            std::cout << "uh\n";
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

    void Sequential::setLossFunction(utils::loss::LossFunction new_loss_function)
    {
        loss_function = new_loss_function;
    }

    cum::Matrix Sequential::forward(const cum::Matrix& input)
    {
        topology[0]->forward(input);

        for(size_t i = 1 ; i < topology.size() ; ++i)
        {
            topology[i]->forward(topology[i - 1]->Outputs());
        }

        return topology.back()->Outputs();
    }

    void Sequential::backward(const cum::Matrix& d_output)
    {
        cum::Matrix curr_gradient = d_output;
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 3)
            std::cout << "\t\t\t" << "layer output gradient: " << yann::utils::logs::matrixToString(curr_gradient.transpose()) << '\n';
        #endif
        for(size_t i = topology.size() - 1 ; i > 0 ; --i)
        {
            // if(topology[i]->layerType() == layers::LAYER_TYPE::INPUT)
            //     continue;

            curr_gradient = topology[i]->backward(curr_gradient);
            #if defined(ENABLE_DEBUG_OUTPUT)     
            if(runtime_config::verbosity_level() >= 3)
                std::cout << "\t\t\t" << "layer " << i << " gradient: " << utils::logs::matrixToString(curr_gradient.transpose()) << '\n';
            #endif
        }        
        #if defined(ENABLE_DEBUG_OUTPUT)
        // if(runtime_config::verbosity_level() >= 3)
        // {
        //     std::cout << "\t\t\t" << "layer 0 gradient: " << utils::logs::matrixToString(curr_gradient.transpose()) << '\n';
        // }
        #endif
    }

    void Sequential::fit(const cum::Matrix& X, const cum::Matrix& Y, cum::cumeric_t rate, size_t epochs)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 1)
            std::cout << "Started training for " << epochs << " epochs...\n";
        #endif
        for(size_t epoch = 0 ; epoch < epochs ; epoch++)
        {
            // Eigen::PermutationMatrix<Eigen::Dynamic> perm(X.rows());
            // perm.setIdentity();
            // cum::Matrix perm(X.rows(), X.rows());
            // std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
            // std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(),
                        // std::mt19937(std::random_device{}()));
            // cum::Matrix X_shuffled = perm * X;
            // cum::Matrix Y_shuffled = perm * Y;
            cum::cummulative_t totalLoss = 0;

            #if defined(ENABLE_DEBUG_OUTPUT)
            if(runtime_config::verbosity_level() >= 1)
                std::cout << "\t" << "Epoch " << epoch << "\n";
            #endif
            for(int i = 0 ; i < X.rows() ; i++)
            {
                #if defined(ENABLE_DEBUG_OUTPUT)
                if(runtime_config::verbosity_level() >= 2)
                    std::cout << "\t\t" << "Sample " << i << "\n";
                #endif

                // cum::Vector x = math_api::matrixTranspose(math_api::matrixRow(X, i));
                // cum::Vector y = math_api::matrixTranspose(math_api::matrixRow(Y, i)); // Todo: check what is shuffling
                // cum::Matrix x = X.row(i).transpose();
                // cum::Matrix y = Y.row(i).transpose();

                cum::Matrix x = X.row(i).transpose();
                cum::Matrix y = Y.row(i).transpose();

                #if defined(ENABLE_DEBUG_OUTPUT)
                if(runtime_config::verbosity_level() >= 2) {
                    // std::cout << "\t\t" << "input: "  << x << '\n';
                    // std::cout << "\t\t" << "target: " << math_api::matrixTranspose(y) << '\n';
                }
                #endif

                cum::Matrix result = this->forward(x);
                #if defined(ENABLE_DEBUG_OUTPUT)
                if(runtime_config::verbosity_level() >= 2)
                {
                    // std::cout << "\t\t" << "resutl: " << math_api::matrixTranspose(result) << '\n';
                    std::cout << "\t\t" << "Computing loss and gradient...\n";
                }
                #endif
                utils::loss::LossType error = utils::loss::computeLoss(result, y, this->loss_function);

                cum::Matrix gradient = error.gradient;

                #if defined(ENABLE_DEBUG_OUTPUT)
                if(runtime_config::verbosity_level() >= 2)
                    std::cout << "\t\t" << "Performing backpropagation...\n";
                #endif

                this->backward(gradient);

                #if defined(ENABLE_DEBUG_OUTPUT)
                if(runtime_config::verbosity_level() >= 2)
                    std::cout << "\t\t" << "Updating parameters...\n";
                #endif

                this->updateParams(rate);
                totalLoss += error.loss;
            }
            // totalLoss /= X.rows();
            cum::cumeric_t avarageLoss = totalLoss / X.rows();

            #if defined(ENABLE_DEBUG_OUTPUT)
            if(runtime_config::verbosity_level() >= 1) {
                std::cout << "\t" << "Avarage epoch loss: " << avarageLoss << '\n';
                std::cout << "\t" << "Total epoch loss: " << totalLoss << '\n';
            }
            #endif
        }
    }

    void Sequential::fit(const cum::Matrix& X, const cum::Matrix& Y, optimizers::OptimizerBase* optimizer, size_t epochs)
    {

    }

    void Sequential::updateParams(cum::cumeric_t rate)
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

    auto Sequential::getLayer(size_t layer) const -> LayerPtr
    {
        // return std::move(topology[layer]);
    }

    auto Sequential::getTopology() const -> Topology
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
}