/**
 * @file OptimizerBase.hpp
 * @author jaro
 * @brief Base class for numerical optimization algorithms in YANN.
 * @date 2026-07-17
 */

#ifndef YANN_OPTIMIZERBASE_HPP
#define YANN_OPTIMIZERBASE_HPP

#include <vector>
#include <memory>

#include "Parameter.hpp"

namespace yann::optimizers
{
    /**
     * @class OptimizerBase
     * @brief Abstract base class for parameter optimization strategies.
     *
     * Provides interfaces for updating model parameters using computed gradients,
     * scaling gradients, and clearing accumulated gradients.
     */
    class OptimizerBase
    {
    public:
        /**
         * @brief Construct an optimizer with a specified learning rate.
         * @param learning_rate Learning rate step size for parameter updates.
         */
        explicit OptimizerBase(cum::cumeric_t learning_rate) : learning_rate(learning_rate) {}

        virtual ~OptimizerBase() = default;

        /**
         * @brief Perform an optimization step on a single Parameter object.
         *
         * Pure virtual method implementing the core parameter update algorithm.
         *
         * @param param Parameter instance containing values and gradients.
         */
        virtual void step(Parameter& param) = 0;

        /**
         * @brief Perform an optimization step on a collection of trainable parameters.
         *
         * Iterates over all non-null parameters and delegates to step(Parameter&).
         * Derived optimizers may override this to implement batched / multi-tensor kernels.
         *
         * @param params Vector of pointers to Parameter instances to update.
         */
        virtual void step(std::vector<Parameter*>& params);

        /**
         * @brief Scale gradients across a vector of parameters by a scalar multiplier.
         * @param params Vector of Parameter pointers whose gradients will be scaled.
         * @param scalar Multiplicative scaling factor.
         */
        virtual void scale_grads(std::vector<Parameter*>& params, cum::cumeric_t scalar);

        /**
         * @brief Reset all accumulated gradients to zero for a vector of parameters.
         * @param params Vector of Parameter pointers whose gradients will be cleared.
         */
        virtual void clear_grads(std::vector<Parameter*>& params);

        cum::cumeric_t learning_rate; /**< Learning rate hyperparameter. */
    };

    /**
     * @brief Type alias for a unique pointer to an OptimizerBase instance.
     */
    using Optimizer = std::unique_ptr<OptimizerBase>;
} // yann

#endif //YANN_OPTIMIZERBASE_HPP
