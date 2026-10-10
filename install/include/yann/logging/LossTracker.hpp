//
// Created by jaro on 7/27/26.
//

#ifndef YANN_LOSSTRACKER_HPP
#define YANN_LOSSTRACKER_HPP

#include "yann/logging/ITrainingCallback.hpp"

namespace yann::logging
{
    class LossTracker : public ITrainingCallback
    {
    public:
        void beforeEpoch(TrainingContext& context) override;
        void afterEpoch(TrainingContext& context) override;

        std::vector<cum::cumeric_t> getLossHistory() const;

    private:
        std::vector<cum::cumeric_t> loss_history;
    };
}

#endif //YANN_LOSSTRACKER_HPP
