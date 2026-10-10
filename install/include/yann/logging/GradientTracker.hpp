//
// Created by jaro on 8/28/26.
//

#ifndef YANN_GRADIENTTRACKER_HPP
#define YANN_GRADIENTTRACKER_HPP

#include "ITrainingCallback.hpp"

namespace yann::logging
{
    class GradientTracker : public ITrainingCallback
    {
    public:
        struct LayerHistory
        {
            std::vector<cum::cumeric_t> weights;
            std::vector<cum::cumeric_t> biases;
        };

        void beforeEpoch(TrainingContext& context) override;
        void afterEpoch(TrainingContext& context) override;
        void afterBackprop(TrainingContext& context) override;

        std::vector<LayerHistory> getLayersHistory() const;
        LayerHistory getLayerHistory(std::size_t index) const;

        enum class tracking_mode { average_absolute_value, sum, squared_norm };
        tracking_mode get_mode();
        void set_mode(tracking_mode mode);
    private:
        tracking_mode mode_ = tracking_mode::average_absolute_value;

        std::vector<LayerHistory> history;
    };
}

#endif //YANN_GRADIENTTRACKER_HPP
