//
// Created by jaro on 7/27/26.
//

#include "yann/logging/LossTracker.hpp"

#include "Sequential.hpp"

namespace yann::logging
{
    void LossTracker::beforeEpoch(TrainingContext& context)
    {
        // loss_history.clear();
    }

    void LossTracker::afterEpoch(TrainingContext& context)
    {
        loss_history.push_back(context.loss);
    }

    std::vector<cum::cumeric_t> LossTracker::getLossHistory() const
    {
        return loss_history;
    }
} // yann