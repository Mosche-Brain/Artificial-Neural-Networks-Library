//
// Created by jaro on 8/28/26.
//

#include "yann/logging/GradientTracker.hpp"

#include "Sequential.hpp"

namespace yann::logging
{
    void GradientTracker::beforeEpoch(TrainingContext& context)
    {
        if (history.size() != context.model.getLayersCount())
        {
            history.resize(context.model.getLayersCount());
        }
    }

    void GradientTracker::afterEpoch(TrainingContext& context)
    {
    }

    void GradientTracker::afterBackprop(TrainingContext& context)
    {
        if (history.size() != context.model.getLayersCount())
        {
            history.resize(context.model.getLayersCount());
        }

        const auto& topology = context.model.getTopology();
        for (std::size_t i = 1; i < topology.size(); ++i)
        {
            switch (mode_)
            {
                case tracking_mode::average_absolute_value:
                    history[i].biases.push_back(topology[i]->BiasesGrad().amean());
                    history[i].weights.push_back(topology[i]->WeightsGrad().amean());
                    break;

                case tracking_mode::sum:
                case tracking_mode::squared_norm:
                    break;
            }
        }
    }

    std::vector<GradientTracker::LayerHistory> GradientTracker::getLayersHistory() const
    {
        return history;
    }

    GradientTracker::LayerHistory GradientTracker::getLayerHistory(std::size_t index) const
    {
        return  history[index];
    }
}
