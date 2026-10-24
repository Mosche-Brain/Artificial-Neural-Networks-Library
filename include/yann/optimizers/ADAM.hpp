/**
 * @file ADAM.hpp
 * @author jaro
 * @brief Adam (Adaptive Moment Estimation) optimizer implementation.
 * @date 2026-07-31
 */

#ifndef YANN_ADAM_HPP
#define YANN_ADAM_HPP

#include <unordered_map>
#include <memory>
#include <vector>

#include "yann/optimizers/OptimizerBase.hpp"

namespace yann::optimizers
{
    /**
     * @class ADAM
     * @brief Adaptive Moment Estimation (Adam) optimizer.
     *
     * Tracks exponentially decaying averages of past gradients (first moment)
     * and past squared gradients (second moment):
     * @f[ m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t @f]
     * @f[ v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 @f]
     * with bias-corrected estimates used for parameter updates.
     */
    class ADAM : public OptimizerBase
    {
    public:
        /**
         * @brief Construct an ADAM optimizer.
         * @param learning_rate Learning rate step size @f$\eta@f$.
         * @param b1 Exponential decay rate for first moment estimates @f$\beta_1@f$ (default: 0.9).
         * @param b2 Exponential decay rate for second moment estimates @f$\beta_2@f$ (default: 0.999).
         */
        ADAM(cum::cumeric_t learning_rate, cum::cumeric_t b1 = 0.9, cum::cumeric_t b2 = 0.999);

        /**
         * @brief Perform an optimization step on a single Parameter instance.
         * @param param Parameter instance to update.
         */
        void step(Parameter& param) override;

        /**
         * @brief Perform an optimization step across multiple Parameter instances.
         * @param params Vector of Parameter pointers to update.
         */
        void step(std::vector<Parameter*>& params) override;

        /**
         * @brief Factory method creating a unique pointer to an ADAM optimizer.
         * @param learning_rate Learning rate step size.
         * @param b1 First moment decay coefficient (default: 0.9).
         * @param b2 Second moment decay coefficient (default: 0.999).
         * @return std::unique_ptr<ADAM> Newly created optimizer.
         */
        static std::unique_ptr<ADAM> create(cum::cumeric_t learning_rate, cum::cumeric_t b1 = 0.9, cum::cumeric_t b2 = 0.999);

    private:
        // std::size_t current_step;
        cum::cumeric_t b1; /**< First moment decay factor beta_1. */
        cum::cumeric_t b2; /**< Second moment decay factor beta_2. */

        std::unordered_map<Parameter*, cum::Tensor> momentum;         /**< First moment buffers. */
        std::unordered_map<Parameter*, cum::Tensor> second_momentum;  /**< Second moment buffers. */
        std::unordered_map<Parameter*, cum::dim_t> steps_count;       /**< Per-parameter update step counter for bias correction. */
    };
} // yann

#endif //YANN_ADAM_HPP
