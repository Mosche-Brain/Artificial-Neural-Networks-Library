/**
 * @file CategoricalCrossEntropy.hpp
 * @author jaro
 * @brief Categorical Cross-Entropy loss function for multi-class classification.
 * @date 2026-09-29
 */

#ifndef YANN_CATEGORICALCROSSENTROPY_HPP
#define YANN_CATEGORICALCROSSENTROPY_HPP

#include "yann/loss/LossBase.hpp"

namespace yann::loss
{
    /**
     * @class CategoricalCrossEntropy
     * @brief Computes categorical cross-entropy loss between probability distributions and targets.
     *
     * Evaluates:
     * @f[
     * L = -\sum_k y_k \log(p_k)
     * @f]
     * Supports both probabilities and raw logits input.
     */
    class CategoricalCrossEntropy : public LossBase
    {
    public:
        /**
         * @brief Construct CategoricalCrossEntropy loss.
         * @param from_logits If true, predictions are treated as unnormalized logits rather than probabilities.
         */
        CategoricalCrossEntropy(bool from_logits = false);

        /**
         * @brief Factory method creating a unique instance of CategoricalCrossEntropy.
         * @param from_logits If true, predictions are treated as raw logits.
         * @return std::unique_ptr to a new CategoricalCrossEntropy instance.
         */
        static std::unique_ptr<CategoricalCrossEntropy> create(bool from_logits = false);

    protected:
        /**
         * @brief Computes sum of categorical cross-entropy loss and its unscaled gradient.
         * @param predicted Predicted class probabilities or logits tensor.
         * @param target Ground-truth targets (e.g. one-hot encoded vectors).
         */
        void compute_sum(const cum::Tensor& predicted, const cum::Tensor& target) override;

    private:
        bool from_logits_;
    };
} // yann::loss

#endif //YANN_CATEGORICALCROSSENTROPY_HPP
