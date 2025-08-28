#include "Sequential.hpp"

namespace SNN::Models
{
    Sequential::Sequential()
    {

    }

    Sequential::Sequential(std::vector<std::unique_ptr<Layers::LayerBase>> topology)
    {
        topology = std::move(topology);

        for(int i = 1 ; i < topology.size() ; i++)
        {
               
        }
    }

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
        for(int i = 1 ; i < topology.size() ; ++i)
        {
            topology[i]->forward(topology[i - 1]->Outputs());
        }

        return topology.back()->Outputs();
    }
}