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

namespace cum { class Matrix; }

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

        // virtual void step(cum::Matrix& params, cum::Matrix& grad) {};

        /**
         * @brief Perform an optimization step on raw matrix parameters and gradients.
         * @param params Parameter matrix to update in-place.
         * @param grad Gradient matrix.
         */
        virtual void step(cum::Matrix& params, cum::Matrix& grad) {};

        /**
         * @brief Perform an optimization step on raw tensor parameters and gradients.
         * @param params Parameter tensor to update in-place.
         * @param grad Gradient tensor.
         */
        virtual void step(cum::Tensor& params, cum::Tensor& grad) {};

        /**
         * @brief Perform an optimization step on a single Parameter object.
         * @param param Parameter instance containing values and gradients.
         */
        virtual void step(Parameter& param) {};

        /**
         * @brief Perform an optimization step on a collection of trainable parameters.
         *
         * Pure virtual method implemented by specific optimizer algorithms.
         *
         * @param params Vector of pointers to Parameter instances to update.
         */
        virtual void step(std::vector<Parameter*>& params) = 0;

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
