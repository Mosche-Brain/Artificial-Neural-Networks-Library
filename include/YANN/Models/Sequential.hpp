#pragma once

#include <initializer_list>
#include <memory>
#include <vector>

#include "Layers/Layers.hpp"
#include "Utility/loss.hpp"
#include "math_api.hpp"

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
        
        matrix_t forward(const matrix_t& input);
        
        void fit(const matrix_t& X, const matrix_t& Y, numeric_t rate, size_t epochs);
        void addLayer(LayerPtr layer);
        void clear();

        /* Getters */
        matrix_t getWeights(size_t layer) const;
        matrix_t getBiases(size_t layer) const;
        Utils::activation_t getActivation(size_t layer) const;
        auto getTopology() const -> Topology;
        auto getLayer(size_t layer) const -> LayerPtr;
        size_t getLayersCount() const;
        
    protected:
        void backward(const matrix_t& d_output);
        void updateParams(numeric_t rate);

        Topology topology;
        Utils::loss::LossFunction loss_function = Utils::loss::LossFunction::mse;
    };
}