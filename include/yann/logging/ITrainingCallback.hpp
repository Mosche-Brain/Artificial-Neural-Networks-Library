//
// Created by jaro on 7/27/26.
//

#pragma once

#include <vector>

#include <cum/cum.hpp>

#include "yann/Parameter.hpp"

namespace yann::models
{
    class Sequential;
}

namespace yann::logging
{
    struct TrainingContext
    {
        models::Sequential& model;
        cum::cumeric_t loss;
        size_t epoch;
        size_t batch;

        TrainingContext(models::Sequential& model, cum::cumeric_t loss, size_t epoch, size_t batch)
            : model(model), loss(loss), epoch(epoch), batch(batch)
        {}
    };

    class ITrainingCallback
    {
    public:
        virtual ~ITrainingCallback() = default;
        virtual void beforeEpoch(TrainingContext& context) = 0;
        virtual void afterEpoch(TrainingContext& context) = 0;
        virtual void afterBackprop(TrainingContext& context) {};
    };

} // yann

