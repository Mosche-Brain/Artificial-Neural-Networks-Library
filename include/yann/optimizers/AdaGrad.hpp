/**
 * @file AdaGrad.hpp
 * @brief AdaGrad (Adaptive Gradient Algorithm) optimizer implementation.
 */

#pragma once

#include <unordered_map>
#include <memory>
#include <vector>

#include "yann/optimizers/OptimizerBase.hpp"

namespace yann::optimizers
{
    /**
     * @class AdaGrad
     * @brief Adaptive gradient optimizer that scales learning rates inversely proportional
     * to the square root of historical squared gradient sums:
     * @f[ G \leftarrow G + (\nabla_\theta L)^2 @f]
     * @f[ \theta \leftarrow \theta - \frac{\eta}{\sqrt{G} + \epsilon} \nabla_\theta L @f]
     */
    class AdaGrad : public OptimizerBase
    {
    public:
        /**
         * @brief Construct an AdaGrad optimizer.
         * @param rate Initial learning rate @f$\eta@f$.
         * @param eps Small numerical stability constant @f$\epsilon@f$ (default: 1e-8).
         */
        AdaGrad(cum::cumeric_t rate, cum::cumeric_t eps = 1e-8);

        /**
         * @brief Update a single Parameter instance using accumulated squared gradients.
         * @param param Parameter instance to update.
         */
        void step(Parameter& param) override;

        /**
         * @brief Update a collection of parameters.
         * @param params Vector of Parameter pointers to update.
         */
        void step(std::vector<Parameter*>& params) override;

        /**
         * @brief Factory method creating a unique pointer to an AdaGrad optimizer.
         * @param rate Learning rate @f$\eta@f$.
         * @param eps Numerical stability constant @f$\epsilon@f$.
         * @return std::unique_ptr<AdaGrad> Newly created optimizer.
         */
        std::unique_ptr<AdaGrad> create(cum::cumeric_t rate, cum::cumeric_t eps = 1e-8);

    private:
        cum::cumeric_t eps;                                     /**< Numerical stability epsilon. */
        std::unordered_map<Parameter*, cum::Tensor> sum_sq_g;   /**< Historical sum of squared gradients. */
    };
}
