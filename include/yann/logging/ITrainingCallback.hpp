/**
 * @file ITrainingCallback.hpp
 * @author jaro
 * @brief Training callback interfaces and execution context for model training observation.
 * @date 2026-07-27
 */

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
    /**
     * @struct TrainingContext
     * @brief Context passed to training callbacks during various phases of training.
     *
     * Provides mutable references to current model, loss, epoch index, and batch index.
     */
    struct TrainingContext
    {
        models::Sequential& model; /**< Reference to the model being trained. */
        cum::cumeric_t& loss;      /**< Current loss value. */
        cum::dim_t& epoch;         /**< Current epoch index. */
        cum::dim_t& batch;         /**< Current batch index within the epoch. */

        /**
         * @brief Construct training context with references to active training state.
         * @param model Reference to the Sequential model.
         * @param loss Reference to the active scalar loss.
         * @param epoch Reference to the current epoch index.
         * @param batch Reference to the current batch index.
         */
        TrainingContext(models::Sequential& model, cum::cumeric_t& loss, cum::dim_t& epoch, cum::dim_t& batch)
            : model(model), loss(loss), epoch(epoch), batch(batch)
        {}
    };

    /**
     * @class ITrainingCallback
     * @brief Interface for observing and reacting to events during model training.
     */
    class ITrainingCallback
    {
    public:
        virtual ~ITrainingCallback() = default;

        /**
         * @brief Hook called immediately prior to starting a new epoch.
         * @param context Active training state context.
         */
        virtual void beforeEpoch(TrainingContext& context) = 0;

        /**
         * @brief Hook called immediately after finishing an epoch.
         * @param context Active training state context.
         */
        virtual void afterEpoch(TrainingContext& context) = 0;

        /**
         * @brief Hook called immediately after backward propagation completes for a batch.
         * @param context Active training state context.
         */
        virtual void afterBackprop(TrainingContext& context) {};
    };

} // yann
