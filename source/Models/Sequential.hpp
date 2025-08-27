#pragma once

#include <Eigen/Dense>

#include <memory>
#include <list>

#include "Layers/Layers.hpp"

namespace SNN::Models
{
    class Sequential
    {
    public:
        Sequential();
        Sequential(std::list<Layers::LayerBase> topology);

        void addLayer(const Layers::LayerBase& layer);


        void forward(Eigen::MatrixXf input);
    
    protected:
        std::list<std::unique_ptr<Layers::LayerBase>> topology;
    };
    
}