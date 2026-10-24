/**
 * @file BinaryCrossEntropy.hpp
 * @author jaro
 * @brief Binary Cross-Entropy loss function implementation for binary classification tasks.
 * @date 2026-07-31
 */

#ifndef YANN_BINARYCROSSENTROPY_HPP
#define YANN_BINARYCROSSENTROPY_HPP

#include "yann/loss/LossBase.hpp"
#include <memory>

namespace yann::loss
{
    /**
     * @class BinaryCrossEntropy
     * @brief Computes binary cross-entropy loss between predictions and binary targets.
     *
     * Evaluates:
     * @f[
     * L = -\sum [y \log(p) + (1 - y) \log(1 - p)]
     * @f]
     */
    class BinaryCrossEntropy : public LossBase
    {
    public:
        /**
         * @brief Factory method creating a unique instance of BinaryCrossEntropy.
         * @return std::unique_ptr to a new BinaryCrossEntropy loss object.
         */
        static std::unique_ptr<LossBase> create();

    protected:
        /**
         * @brief Computes sum of binary cross-entropy loss and its unscaled gradient.
         * @param predicted Predicted probabilities tensor (values in range [0, 1]).
         * @param target Ground-truth binary labels tensor (values 0 or 1).
         */
        void compute_sum(const cum::Tensor& predicted, const cum::Tensor& target) override;
    };

} // yann

#endif //YANN_BINARYCROSSENTROPY_HPP
