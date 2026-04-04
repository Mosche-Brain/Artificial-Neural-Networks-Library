#pragma once

#include <initializer_list>
#include <memory>
#include <vector>

#include "Layers/Layers.hpp"
#include "Utility/loss.hpp"
#include "math_api.hpp"

namespace YANN::Models
{
    using Topology = std::vector<std::unique_ptr<Layers::LayerBase>>;
    using LayerPtr = std::unique_ptr<Layers::LayerBase>;
    class Sequential
    {
    public:
        Sequential() ;
        // Sequential(std::vector<std::unique_ptr<Layers::LayerBase>> newTopology);
        Sequential(std::initializer_list<std::unique_ptr<Layers::LayerBase>> newTopology);

        void addLayer(LayerPtr layer);

        matrix_t forward(matrix_t input);
        void backward(const matrix_t& d_output);

        void fit(const matrix_t& X, const matrix_t& Y, numeric_t rate, size_t epochs);
        void updateParams(numeric_t rate);

        matrix_t getWeights(size_t layer);
        matrix_t getBiases(size_t layer);
        Utils::activation_t getActivation(size_t layer);
        Topology getTopology();
        LayerPtr getLayer(size_t layer);
        size_t getLayersCount();

    protected:
        Topology topology;
        Utils::loss::LossFunction loss_function = Utils::loss::LossFunction::mse;
    };
}