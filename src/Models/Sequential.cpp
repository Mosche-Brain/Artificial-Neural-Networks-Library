#include "Sequential.hpp"

#include <iostream>

namespace ANN::Models
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
            // std::cout << "bach\n";
            topology[i]->forward(topology[i - 1]->Outputs());
        }

        return topology.back()->Outputs();
    }

    void Sequential::backward(const Eigen::MatrixXf& d_output)
    {
        Eigen::MatrixXf curr_gradient = d_output;
        // std::cout << "layer " <<  << " gradient:\n" << curr_gradient << '\n';
        std::cout << "layer output gradient:\n" << curr_gradient << '\n';
        for(int i = topology.size() - 1 ; i >= 0 ; --i)
        {
            curr_gradient = topology[i]->backward(curr_gradient);
            std::cout << "layer " << i << " gradient:\n" << curr_gradient << '\n';
        }        
        // std::cout << "layer 0 gradient:\n" << curr_gradient << '\n';
    }

    void Sequential::fit(const Eigen::MatrixXf& X, const Eigen::MatrixXf& Y, float_t rate, int epochs)
    {
        for(int epoch = 0 ; epoch < epochs ; epoch++)
        {
            Eigen::PermutationMatrix<Eigen::Dynamic> perm(X.rows());
            perm.setIdentity();
            std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());

            Eigen::MatrixXf X_shuffled = perm * X;
            Eigen::MatrixXf Y_shuffled = perm * Y;
            
            float_t totalLoss = 0;
            // std::cout << "epoch " << epoch << " started\n"; 
            std::cout << "================Epoch " << epoch << "================\n";
            for(int i = 0 ; i < X.rows() ; i++)
            {
                std::cout << "================Sample " << i << "================\n";
                Eigen::VectorXf x = X.row(i).transpose();
                Eigen::VectorXf y = Y.row(i).transpose();

                std::cout << "input: "  << x << '\n';
                std::cout << "target: " << y.transpose() << '\n';

                Eigen::VectorXf result = this->forward(x);
                std::cout << "resutl: " << result.transpose() << '\n';

                std::cout << "=============Computing Loss=============\n";
                Utils::loss::LossType error = Utils::loss::computeLoss(result, y, this->loss_function);
                
                Eigen::VectorXf gradient = error.gradient;
                
                std::cout << "gradient:\n" << gradient << '\n';
                std::cout << "loss: " << error.loss << '\n';
                
                std::cout << "=============Backpropagation=============\n";
                this->backward(gradient);
                std::cout << "=============Updating Params=============\n";
                
                this->updateParams(rate);
                totalLoss += error.loss;
            }
            // totalLoss /= X.rows();
            float_t avarageLoss = totalLoss / X.rows();
            
            std::cout << "Avarage epoch loss: " << avarageLoss << '\n';
            std::cout << "Total epoch loss: " << totalLoss << '\n';
        }
    }

    void Sequential::updateParams(float_t rate)
    {
        for(int i = 0 ; i < topology.size() ; i++)
        {
            if(topology[i]->layerType() != Layers::LAYER_TYPE::INPUT)
                topology[i]->update_weights(rate);
        }
    }

    Eigen::MatrixXf Sequential::getWeights(int layer_idx)
    {
        return topology[layer_idx]->Weights();
    }

}