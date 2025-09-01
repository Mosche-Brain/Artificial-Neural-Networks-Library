#include "Sequential.hpp"

#include <iostream>

namespace SNN::Models
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

        // topology[0]->initParameters(topology[0]->size(), 1);
        for(size_t i = 1 ; i < topology.size() ; i++)
        {
            int previous_layer_size = topology[i - 1]->size();
            int  current_layer_size = topology[  i  ]->size();

            topology[i]->initParameters(current_layer_size, previous_layer_size);
        }

        for(size_t i = 0 ; i < topology.size() ; i++)
        {
            std::cout << "layer " << i << ": " << "weights " << topology[i]->Weights().rows() << "x" << topology[i]->Weights().cols() << " biases " << topology[i]->Biases().size() << '\n';
        }
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

    Eigen::MatrixXf Sequential::forward(Eigen::MatrixXf input)
    {
        topology[0]->forward(input);

        for(int i = 1 ; i < topology.size() ; ++i)
        {
            std::cout << "bach\n";
            topology[i]->forward(topology[i - 1]->Outputs());
        }

        return topology.back()->Outputs();
    }

    void Sequential::backward(const Eigen::MatrixXf& d_output)
    {
        Eigen::MatrixXf curr_gradient = d_output;
        for(int i = topology.size() - 1 ; i >= 1 ; --i)
        {
            std::cout << "brrr\n";
            curr_gradient = topology[i]->backward(curr_gradient);
        }        
    }

    void Sequential::fit(const Eigen::MatrixXf& X, const Eigen::MatrixXf& Y, float_t rate, int epochs)
    {
        for(int epoch = 0 ; epoch < epochs ; epoch++)
        {
            float_t epoch_loss = 0;
            for(int i = 0 ; i < X.rows() ; i++)
            {
                Eigen::VectorXf x = X.row(i).transpose();
                Eigen::VectorXf y = Y.row(i).transpose();

                Eigen::VectorXf result = this->forward(x);

                Utils::loss::LossType loss = Utils::loss::computeLoss(result, y, this->loss_function);

                Eigen::MatrixXf gradient = loss.gradient;

                this->backward(gradient);
                    std::cout << "back" << '\n';
                this->updateParams(rate);
                    std::cout << "update" << '\n';
                
                epoch_loss += loss.loss;
            }
            epoch_loss /= X.rows();

            std::cout << "Loss: " << epoch_loss << '\n';
        }
    }

    void Sequential::updateParams(float_t rate)
    {
        for(int i = 0 ; i < topology.size() ; i++)
        {
            topology[i]->update_weights(rate);
        }
    }
}