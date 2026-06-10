#include "Sequential.hpp"

#include <algorithm>
#include <random>

#if defined(ENABLE_DEBUG_OUTPUT)
    #include <iostream>
    #include "Utility/logs.hpp"
#endif

#include "runtime_config.hpp"
namespace YANN::Models
{
    Sequential::Sequential()
    {

    }

    Sequential::Sequential(std::initializer_list<std::unique_ptr<Layers::LayerBase>> newTopology)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 2)
            std::cout << "Initializing Sequential model with " << newTopology.size() << " layers...\n";
        #endif

        topology.reserve(newTopology.size()); 
        for(auto& ptr : newTopology) 
        {
            std::cout << "oh\n";
            topology.push_back(std::move(const_cast<std::unique_ptr<Layers::LayerBase>&>(ptr)));
        }
        
        std::cout << "ah\n";
        topology[0]->initParameters(topology[0]->size(), 1);
        for(size_t i = 1 ; i < topology.size() ; i++)
        {
            std::cout << "uh\n";
            int previous_layer_size = topology[i - 1]->size();
            int  current_layer_size = topology[  i  ]->size();

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

    cum::Matrix Sequential::forward(const cum::Matrix& input)
    {
        topology[0]->forward(input);
        std::cout << "layer 0 output dimensions " << Utils::logs::show_matrix_dimensions(topology[0]->Outputs()) << '\n';

        for(size_t i = 1 ; i < topology.size() ; ++i)
        {
            std::cout << "bach\n";
            topology[i]->forward(topology[i - 1]->Outputs());
            std::cout << "layer " << i << " output dimensions " << Utils::logs::show_matrix_dimensions(topology[i]->Outputs()) << '\n';
        }

        return topology.back()->Outputs();
    }

    void Sequential::backward(const cum::Matrix& d_output)
    {
        cum::Matrix curr_gradient = d_output;
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 3)
            std::cout << "\t\t\t" << "layer output gradient: " << YANN::Utils::logs::matrixToString(curr_gradient) << '\n';
        #endif
        for(size_t i = topology.size() - 1 ; i > 0 ; --i)
        {
            std::cout << "layer type: " << topology[i]->layerType() << '\n';
            curr_gradient = topology[i]->backward(curr_gradient);
            #if defined(ENABLE_DEBUG_OUTPUT)     
            if(runtime_config::verbosity_level() >= 3)
                std::cout << "\t\t\t" << "layer " << i << " gradient: " << Utils::logs::matrixToString(curr_gradient) << '\n';
                // std::cout << "layer type: " << topology[i]->layerType() << '\n';
            #endif
        }        
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::verbosity_level() >= 3)
        {
            std::cout << "\t\t\t" << "layer 0 gradient: " << Utils::logs::matrixToString(curr_gradient) << '\n';
        }
        #endif
    }

    void Sequential::fit(const cum::Matrix& X, const cum::Matrix& Y, cum::cumeric_t rate, size_t epochs)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 1)
            std::cout << "Starting training for " << epochs << " epochs...\n";
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
            


            cum::cumeric_t totalLoss = 0_c;



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

                cum::Matrix x = X.row(i);
                cum::Matrix y = Y.row(i);

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
                Utils::loss::LossType error = Utils::loss::computeLoss(result, y, this->loss_function);
                
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
            if(runtime_config::DEBUG_VEBOSITY >= 1) {
                std::cout << "\t" << "Avarage epoch loss: " << avarageLoss << '\n';
                std::cout << "\t" << "Total epoch loss: " << totalLoss << '\n';
            }
            #endif
        }
    }

    void Sequential::updateParams(cum::cumeric_t rate)
    {
        for(size_t i = 0 ; i < topology.size() ; i++)
        {
            if(topology[i]->layerType() != Layers::LAYER_TYPE::INPUT)
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
}