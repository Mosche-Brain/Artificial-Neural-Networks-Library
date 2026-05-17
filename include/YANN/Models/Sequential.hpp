#pragma once

#include <initializer_list>
#include <memory>
#include <vector>

#include "Layers/Layers.hpp"
#include "Utility/loss.hpp"

namespace YANN::Models
{
    class Sequential
    {
    public:
        using Topology = std::vector<std::unique_ptr<Layers::LayerBase>>;
        using LayerPtr = std::unique_ptr<Layers::LayerBase>;

        /* Constructors */
        Sequential() ;
        Sequential(std::initializer_list<LayerPtr> newTopology);

        /* Methods */
        
        cum::Matrix forward(const cum::Matrix& input);
        
        void fit(const cum::Matrix& X, const cum::Matrix& Y, cum::cumeric_t rate, size_t epochs);
        void addLayer(LayerPtr layer);
        void clear();

        /* Getters */
        cum::Matrix& getWeights(size_t layer) const;
        cum::Matrix& getBiases(size_t layer) const;
        cum::functions::activation_t getActivation(size_t layer) const;
        auto getTopology() const -> Topology;
        auto getLayer(size_t layer) const -> LayerPtr;
        size_t getLayersCount() const;
        
    protected:
        void backward(const cum::Matrix& d_output);
        void updateParams(cum::cumeric_t rate);

        Topology topology;
        Utils::loss::LossFunction loss_function = Utils::loss::LossFunction::mse;
    };
}