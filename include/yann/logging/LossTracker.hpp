/**
 * @file LossTracker.hpp
 * @author jaro
 * @brief Training callback for recording loss values across epochs.
 * @date 2026-07-27
 */

#ifndef YANN_LOSSTRACKER_HPP
#define YANN_LOSSTRACKER_HPP

#include "yann/logging/ITrainingCallback.hpp"
#include <vector>

namespace yann::logging
{
    /**
     * @class LossTracker
     * @brief Callback implementation that records loss values after each training epoch.
     */
    class LossTracker : public ITrainingCallback
    {
    public:
        /**
         * @brief Called before an epoch starts (no-op for LossTracker).
         * @param context Active training context.
         */
        void beforeEpoch(TrainingContext& context) override;

        /**
         * @brief Called after an epoch ends; appends current loss to history.
         * @param context Active training context containing current loss.
         */
        void afterEpoch(TrainingContext& context) override;

        /**
         * @brief Retrieve recorded history of epoch losses.
         * @return Vector containing scalar loss values per epoch.
         */
        std::vector<cum::cumeric_t> getLossHistory() const;

    private:
        std::vector<cum::cumeric_t> loss_history; /**< History of recorded epoch loss values. */
    };
}

#endif //YANN_LOSSTRACKER_HPP
