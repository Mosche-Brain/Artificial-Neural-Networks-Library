/**
 * @file MeanSquaredError.hpp
 * @author jaro
 * @brief Mean Squared Error (MSE / L2 loss) implementation.
 * @date 2026-07-31
 */

#ifndef YANN_MEANSQUAREDERROR_HPP
#define YANN_MEANSQUAREDERROR_HPP

#include "yann/loss/LossBase.hpp"
#include "cum/Matrix.hpp"

namespace yann::loss
{
    /**
     * @class MeanSquaredError
     * @brief Computes Mean Squared Error (L2 loss) between predictions and targets.
     *
     * Evaluates:
     * @f[
     * L = \sum (y - \hat{y})^2
     * @f]
     */
    class MeanSquaredError : public LossBase
    {
    public:
        /**
         * @brief Factory method creating a unique instance of MeanSquaredError.
         * @return std::unique_ptr to a new MeanSquaredError instance.
         */
        static std::unique_ptr<LossBase> create();

    protected:
        /**
         * @brief Computes sum of squared errors and unscaled gradient.
         * @param predicted Predicted values tensor.
         * @param target Ground-truth target tensor.
         */
        void compute_sum(const cum::Tensor& predicted, const cum::Tensor& target) override;
    };
} // yann

#endif //YANN_MEANSQUAREDERROR_HPP
