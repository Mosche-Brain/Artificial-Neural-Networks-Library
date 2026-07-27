#pragma once

#include <initializer_list>
#include <memory>
#include <vector>

#include "layers/Layers.hpp"
#include "../loss.hpp"
#include "logging/ITrainingCallback.hpp"
#include "optimizers//OptimizerBase.hpp"

/* # Used conventions
 * - Layer indexing starts from 0
 * - Vectors are 1D matrices
 * - Inputs and results are column vectors
 * - Training data are transposed (every row is a sample, it is transposed in training)
 */

namespace yann::models
{
    class Sequential
    {
    public:
        using Topology = std::vector<std::unique_ptr<layers::LayerBase>>;
        using LayerPtr = std::unique_ptr<layers::LayerBase>;

        /* Constructors */
        Sequential() ;
        Sequential(std::initializer_list<LayerPtr> newTopology);

        /* Methods */
        
        cum::Matrix forward(const cum::Matrix& input);
        
        void fit(const cum::Matrix& X, const cum::Matrix& Y, cum::cumeric_t rate, size_t epochs);
        void fit(const cum::Matrix& X, const cum::Matrix& Y, optimizers::OptimizerBase& optimizer, size_t epochs, std::span<logging::ITrainingCallback*> callbacks = {});
        void addLayer(LayerPtr layer);
        void clear();

        void setLossFunction(loss::LossFunction new_loss_function);

        /* Getters */
        cum::Matrix& getWeights(size_t layer) const;
        cum::Matrix& getBiases(size_t layer) const;
        cum::Matrix& getOutputs(size_t layer) const;
        cum::functions::activation_t getActivation(size_t layer) const;
        auto getTopology() const -> Topology;
        auto getLayer(size_t layer) const -> LayerPtr;
        size_t getLayersCount() const;
        int getLayerSize(size_t layer) const;

        std::vector<Parameter*> parameters();
    protected:
        void backward(const cum::Matrix& d_output);
        void updateParams(cum::cumeric_t rate);

        Topology topology;
        loss::LossFunction loss_function;
    };
}