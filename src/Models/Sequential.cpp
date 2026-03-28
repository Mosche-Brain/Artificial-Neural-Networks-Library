#include "Sequential.hpp"

#include <iostream>
#include <algorithm>
#include <random>

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

        // for(size_t i = 0 ; i < topology.size() ; i++)
        // {
        //     std::cout << "layer " << i << ": " << "weights " << topology[i]->Weights().rows() << "x" << topology[i]->Weights().cols() << " biases " << topology[i]->Biases().size() << '\n';
        // }
    }

    // Sequential::Sequential(std::vector<std::unique_ptr<Layers::LayerBase>> newTopology)
    // {
    //     topology = std::move(newTopology);

    //     for(size_t i = 1 ; i < topology.size() ; i++)
    //     {
               
    //     }
    // }

    void Sequential::addLayer(std::unique_ptr<Layers::LayerBase> layer)
    {
        topology.push_back(std::move(layer));

        if(topology.size() == 1)
        {
            int layerSize = topology.back()->size();
            topology.back()->initParameters(layerSize, 1);
        }
        else
        {
            int layerSize  = topology.back()->size();
            int inputWidth = topology[topology.size() - 1]->size();
            
            topology.back()->initParameters(layerSize, inputWidth );
        }
    }

    matrix_t Sequential::forward(matrix_t input)
    {
        topology[0]->forward(input);

        for(int i = 1 ; i < topology.size() ; ++i)
        {
            // std::cout << "bach\n";
            topology[i]->forward(topology[i - 1]->Outputs());
        }

        return topology.back()->Outputs();
    }

    void Sequential::backward(const matrix_t& d_output)
    {
        matrix_t curr_gradient = d_output;
        // std::cout << "layer " <<  << " gradient:\n" << curr_gradient << '\n';
        // std::cout << "layer output gradient:\n" << curr_gradient << '\n';
        for(int i = topology.size() - 1 ; i >= 0 ; --i)
        {
            curr_gradient = topology[i]->backward(curr_gradient);
            // std::cout << "layer " << i << " gradient:\n" << curr_gradient << '\n';
        }        
        // std::cout << "layer 0 gradient:\n" << curr_gradient << '\n';
    }

    void Sequential::fit(const matrix_t& X, const matrix_t& Y, numeric_t rate, int epochs)
    {
        for(int epoch = 0 ; epoch < epochs ; epoch++)
        {
            Eigen::PermutationMatrix<Eigen::Dynamic> perm(X.rows());
            perm.setIdentity();
            // std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
            std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), 
                        std::mt19937(std::random_device{}()));
            
            matrix_t X_shuffled = perm * X;
            matrix_t Y_shuffled = perm * Y;
            
            numeric_t totalLoss = 0;
            std::cout << "================Epoch " << epoch << "================\n";
            for(int i = 0 ; i < X.rows() ; i++)
            {
                std::cout << "================Sample " << i << "================\n";
                vector_t x = X.row(i).transpose();
                vector_t y = Y.row(i).transpose();

                std::cout << "input: "  << x << '\n';
                std::cout << "target: " << y.transpose() << '\n';

                vector_t result = this->forward(x);
                std::cout << "resutl: " << result.transpose() << '\n';

                std::cout << "=============Computing Loss=============\n";
                Utils::loss::LossType error = Utils::loss::computeLoss(result, y, this->loss_function);
                
                vector_t gradient = error.gradient;
                
                std::cout << "gradient:\n" << gradient << '\n';
                std::cout << "loss: " << error.loss << '\n';
                
                std::cout << "=============Backpropagation=============\n";
                this->backward(gradient);
                std::cout << "=============Updating Params=============\n";
                
                this->updateParams(rate);
                totalLoss += error.loss;
            }
            // totalLoss /= X.rows();
            numeric_t avarageLoss = totalLoss / X.rows();
            
            std::cout << "Avarage epoch loss: " << avarageLoss << '\n';
            // std::cout << "Total epoch loss: " << totalLoss << '\n';
        }
    }

    void Sequential::updateParams(numeric_t rate)
    {
        for(int i = 0 ; i < topology.size() ; i++)
        {
            if(topology[i]->layerType() != Layers::LAYER_TYPE::INPUT)
                topology[i]->update_weights(rate);
        }
    }

    matrix_t Sequential::getWeights(int layer_idx)
    {
        return topology[layer_idx]->Weights();
    }

}