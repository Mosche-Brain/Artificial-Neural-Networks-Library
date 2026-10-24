/**
 * @file MeanAbsoluteError.hpp
 * @author jaro
 * @brief Mean Absolute Error (MAE / L1 loss) implementation.
 * @date 2026-07-31
 */

#ifndef YANN_MEANABSOLUTEERROR_HPP
#define YANN_MEANABSOLUTEERROR_HPP

#include <memory>

#include "LossBase.hpp"
#include "cum/Matrix.hpp"

namespace yann::loss
{
    /**
     * @class MeanAbsoluteError
     * @brief Computes Mean Absolute Error (L1 loss) between predictions and targets.
     *
     * Evaluates:
     * @f[
     * L = \sum |y - \hat{y}|
     * @f]
     */
    class MeanAbsoluteError : public LossBase
    {
    public:
        /**
         * @brief Factory method creating a unique instance of MeanAbsoluteError.
         * @return std::unique_ptr to a new MeanAbsoluteError instance.
         */
        static std::unique_ptr<LossBase> create();

    protected:
        /**
         * @brief Computes sum of absolute differences and unscaled subgradient.
         * @param predicted Predicted values tensor.
         * @param target Ground-truth target tensor.
         */
        void compute_sum(const cum::Tensor& predicted, const cum::Tensor& target) override;
    };
} // yann

#endif //YANN_MEANABSOLUTEERROR_HPP
