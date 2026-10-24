/**
 * @file GradientTracker.hpp
 * @author jaro
 * @brief Training callback for monitoring and logging layer gradient statistics.
 * @date 2026-08-28
 */

#ifndef YANN_GRADIENTTRACKER_HPP
#define YANN_GRADIENTTRACKER_HPP

#include "ITrainingCallback.hpp"
#include <vector>
#include <cstddef>

namespace yann::logging
{
    /**
     * @class GradientTracker
     * @brief Callback that tracks gradient magnitudes (weights and biases) across layers during training.
     */
    class GradientTracker : public ITrainingCallback
    {
    public:
        /**
         * @struct LayerHistory
         * @brief History of tracked gradient metrics for a single layer.
         */
        struct LayerHistory
        {
            std::vector<cum::cumeric_t> weights; /**< Tracked weight gradient metric per measurement point. */
            std::vector<cum::cumeric_t> biases;  /**< Tracked bias gradient metric per measurement point. */
        };

        /**
         * @brief Hook invoked before an epoch starts.
         * @param context Active training context.
         */
        void beforeEpoch(TrainingContext& context) override;

        /**
         * @brief Hook invoked after an epoch concludes.
         * @param context Active training context.
         */
        void afterEpoch(TrainingContext& context) override;

        /**
         * @brief Hook invoked after backpropagation completes for a batch.
         * Computes and appends gradient statistics according to active tracking_mode.
         * @param context Active training context.
         */
        void afterBackprop(TrainingContext& context) override;

        /**
         * @brief Get tracked gradient statistics across all network layers.
         * @return Vector of LayerHistory structures for each layer.
         */
        std::vector<LayerHistory> getLayersHistory() const;

        /**
         * @brief Get gradient history for a specific layer index.
         * @param index Layer index in the model topology.
         * @return LayerHistory structure for the requested layer.
         */
        LayerHistory getLayerHistory(std::size_t index) const;

        /**
         * @enum tracking_mode
         * @brief Aggregation strategy for summarizing gradient tensors into scalar metrics.
         */
        enum class tracking_mode 
        { 
            average_absolute_value, /**< Mean of absolute gradient values. */
            sum,                    /**< Sum of gradient values. */
            squared_norm            /**< L2 squared norm of gradient elements. */
        };

        /**
         * @brief Get the currently active gradient tracking mode.
         * @return Current tracking_mode enum value.
         */
        tracking_mode get_mode();

        /**
         * @brief Set the gradient aggregation mode.
         * @param mode Desired tracking_mode value.
         */
        void set_mode(tracking_mode mode);

    private:
        tracking_mode mode_ = tracking_mode::average_absolute_value; /**< Active aggregation metric mode. */
        std::vector<LayerHistory> history;                          /**< History records per layer. */
    };
}

#endif //YANN_GRADIENTTRACKER_HPP
