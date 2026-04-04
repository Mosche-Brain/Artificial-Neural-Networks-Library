#include "Sequential.hpp"

#include <algorithm>
#include <random>

#if defined(ENABLE_DEBUG_OUTPUT)
    #include <iostream>
#endif

#include "runtime_config.hpp"
namespace YANN::Models
{
    Sequential::Sequential()
    {

    }

    Sequential::Sequential(std::initializer_list<std::unique_ptr<Layers::LayerBase>> newTopology)
    {
        topology.reserve(newTopology.size()); 
        for(auto& ptr : newTopology) 
        {
            topology.push_back(std::move(const_cast<std::unique_ptr<Layers::LayerBase>&>(ptr)));
        }

        topology[0]->initParameters(topology[0]->size(), 1);
        for(size_t i = 1 ; i < topology.size() ; i++)
        {
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

    matrix_t Sequential::forward(matrix_t input)
    {
        topology[0]->forward(input);

        for(size_t i = 1 ; i < topology.size() ; ++i)
        {
            // std::cout << "bach\n";
            topology[i]->forward(topology[i - 1]->Outputs());
        }

        return topology.back()->Outputs();
    }

    void Sequential::backward(const matrix_t& d_output)
    {
        matrix_t curr_gradient = d_output;
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 3)
            std::cout << "\t\t\t" << "layer output gradient: " << math_api::matrixTranspose(curr_gradient) << '\n';
        #endif
        for(size_t i = topology.size() - 1 ; i > 0 ; --i)
        {
            curr_gradient = topology[i]->backward(curr_gradient);
            #if defined(ENABLE_DEBUG_OUTPUT)
            if(runtime_config::DEBUG_VEBOSITY >= 3)
                std::cout << "\t\t\t" << "layer " << i << " gradient: " << math_api::matrixTranspose(curr_gradient) << '\n';
            #endif
        }        
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 3)
        {
            std::cout << "\t\t\t" << "layer 0 gradient: " << math_api::matrixTranspose(curr_gradient) << '\n';
        }
        #endif
    }

    void Sequential::fit(const matrix_t& X, const matrix_t& Y, numeric_t rate, size_t epochs)
    {
        #if defined(ENABLE_DEBUG_OUTPUT)
        if(runtime_config::DEBUG_VEBOSITY >= 1)
            std::cout << "Starting training for " << epochs << " epochs...\n";
        #endif
        for(size_t epoch = 0 ; epoch < epochs ; epoch++)
        {
            Eigen::PermutationMatrix<Eigen::Dynamic> perm(X.rows());
            perm.setIdentity();
            // std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
            std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), 
                        std::mt19937(std::random_device{}()));
            
            matrix_t X_shuffled = perm * X;
            matrix_t Y_shuffled = perm * Y;
            
            numeric_t totalLoss = 0;



            #if defined(ENABLE_DEBUG_OUTPUT)
            if(runtime_config::DEBUG_VEBOSITY >= 1)
                std::cout << "\t" << "Epoch " << epoch << "\n";
            #endif
            for(size_t i = 0 ; i < X.rows() ; i++)
            {
                #if defined(ENABLE_DEBUG_OUTPUT)
                if(runtime_config::DEBUG_VEBOSITY >= 2)
                    std::cout << "\t\t" << "Sample " << i << "\n";
                #endif

                vector_t x = math_api::matrixTranspose(math_api::matrixRow(X, i));
                vector_t y = math_api::matrixTranspose(math_api::matrixRow(Y, i)); // Todo: check what is shuffling

                #if defined(ENABLE_DEBUG_OUTPUT)
                if(runtime_config::DEBUG_VEBOSITY >= 2) {
                    std::cout << "\t\t" << "input: "  << x << '\n';
                    std::cout << "\t\t" << "target: " << math_api::matrixTranspose(y) << '\n';
                }
                #endif

                vector_t result = this->forward(x);
                #if defined(ENABLE_DEBUG_OUTPUT)
                if(runtime_config::DEBUG_VEBOSITY >= 2)
                {
                    std::cout << "\t\t" << "resutl: " << math_api::matrixTranspose(result) << '\n';

                    std::cout << "\t\t" << "Computing loss and gradient...\n";
                }
                #endif                
                Utils::loss::LossType error = Utils::loss::computeLoss(result, y, this->loss_function);
                vector_t gradient = error.gradient;
                

                #if defined(ENABLE_DEBUG_OUTPUT)
                if(runtime_config::DEBUG_VEBOSITY >= 2)
                    std::cout << "\t\t" << "Performing backpropagation...\n";
                #endif

                this->backward(gradient);

                #if defined(ENABLE_DEBUG_OUTPUT)
                if(runtime_config::DEBUG_VEBOSITY >= 2)
                    std::cout << "\t\t" << "Updating parameters...\n";
                #endif              

                this->updateParams(rate);
                totalLoss += error.loss;
            }
            // totalLoss /= X.rows();
            numeric_t avarageLoss = totalLoss / X.rows();
            
            #if defined(ENABLE_DEBUG_OUTPUT)
            if(runtime_config::DEBUG_VEBOSITY >= 1) {
                std::cout << "\t" << "Avarage epoch loss: " << avarageLoss << '\n';
                std::cout << "\t" << "Total epoch loss: " << totalLoss << '\n';
            }
            #endif
        }
    }

    void Sequential::updateParams(numeric_t rate)
    {
        for(size_t i = 0 ; i < topology.size() ; i++)
        {
            if(topology[i]->layerType() != Layers::LAYER_TYPE::INPUT)
                topology[i]->update_weights(rate);
        }
    }

    matrix_t Sequential::getWeights(size_t layer)
    {
        return topology[layer]->Weights();
    }

    matrix_t Sequential::getBiases(size_t layer)
    {
        return topology[layer]->Biases();
    }

    Utils::activation_t Sequential::getActivation(size_t layer)
    {
        return topology[layer]->activation;
    }

    LayerPtr Sequential::getLayer(size_t layer)
    {
        return std::move(topology[layer]);
    }

    Topology Sequential::getTopology()
    {
        return topology;
    }

    size_t Sequential::getLayersCount()
    {
        return topology.size();
    }
}