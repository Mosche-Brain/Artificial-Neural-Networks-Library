/**
 * @file LossBase.hpp
 * @author jaro
 * @brief Base class and definitions for loss functions in YANN.
 * @date 2026-07-31
 */

#ifndef YANN_LOSSBASE_HPP
#define YANN_LOSSBASE_HPP

#include <memory>
#include <stdexcept>

#include "cum/Core.hpp"
#include "cum/Matrix.hpp"
#include "cum/Tensor.hpp"

namespace yann::loss
{
    /**
     * @struct loss_t
     * @brief Container holding the calculated scalar loss and its corresponding gradient tensor.
     */
    typedef struct
    {
        cum::cumeric_t value;  /**< Scalar loss value. */
        cum::Tensor gradient;  /**< Gradient of the loss with respect to predictions. */
    } loss_t;

    /**
     * @enum Reduction
     * @brief Batch reduction mode for loss and its gradient.
     *
     * In both variants, loss is summed across feature dimensions:
     * - BatchMean: Sum over features, averaged over batch dimension (default).
     * - Sum: Sum over both features and batch dimension.
     */
    enum class Reduction
    {
        BatchMean, /**< Sum across features, average across batch (default). */
        Sum        /**< Sum across features and across batch. */
    };

    /**
     * @class LossBase
     * @brief Abstract base class for all neural network loss functions in YANN.
     */
    class LossBase
    {
    public:
        /**
         * @brief Fixed batch dimension index for tensors across the framework (layout [batch, ...]).
         */
        static constexpr cum::dim_t batch_axis = 0;

        virtual ~LossBase() = default;

        /**
         * @brief Validates shapes, invokes compute_sum(), and applies the chosen reduction.
         *
         * @param predicted Predicted tensor output from the model.
         * @param target Ground-truth target tensor matching prediction shape.
         * @param reduction Desired reduction mode (default: Reduction::BatchMean).
         */
        void compute(const cum::Tensor& predicted, const cum::Tensor& target, Reduction reduction = Reduction::BatchMean);

        /**
         * @brief Get const reference to the computed loss structure.
         * @return Const reference to loss_t containing scalar value and gradient tensor.
         */
        const loss_t& result() const { return loss; }

        /**
         * @brief Get the computed scalar loss value.
         * @return Scalar loss value.
         */
        cum::cumeric_t getLoss() const { return loss.value; }

        /**
         * @brief Get the computed gradient tensor.
         * @return Gradient tensor with respect to predictions.
         */
        cum::Tensor getGradient() const { return loss.gradient; }

    protected:
        /**
         * @brief Computes raw unscaled loss and gradient summed over features and batch.
         *
         * Pure virtual method implemented by specific loss subclasses.
         *
         * @param predicted Predicted tensor output from the model.
         * @param target Ground-truth target tensor.
         */
        virtual void compute_sum(const cum::Tensor& predicted, const cum::Tensor& target) = 0;

        loss_t loss; /**< Cached loss value and gradient tensor. */
    };

    /**
     * @brief Convenient type alias for a unique pointer to a LossBase instance.
     */
    using Loss = std::unique_ptr<LossBase>;

} // yann

#endif //YANN_LOSSBASE_HPP
