/**
 * @file Momentum.hpp
 * @author Świadomy Mężczyzna
 * @brief Stochastic Gradient Descent with Momentum optimizer.
 * @date 2026-08-09
 */

#pragma once

#include <unordered_map>
#include <memory>
#include <vector>

#include "yann/optimizers/OptimizerBase.hpp"

namespace yann::optimizers
{
    /**
     * @class Momentum
     * @brief SGD optimizer with momentum velocity accumulation:
     * @f[ v \leftarrow \beta v + (1 - \beta) \nabla_\theta L @f]
     * @f[ \theta \leftarrow \theta - \eta v @f]
     */
    class Momentum : public OptimizerBase
    {
    public:
        /**
         * @brief Construct a Momentum optimizer.
         * @param rate Learning rate @f$\eta@f$.
         * @param beta Momentum coefficient @f$\beta@f$ (default: 0.5).
         */
        Momentum(cum::cumeric_t rate, cum::cumeric_t beta = 0.5);

        /**
         * @brief Perform an optimization step on a single parameter.
         * @param param Parameter instance to update.
         */
        void step(Parameter& param) override;

        /**
         * @brief Perform an optimization step across multiple parameters.
         * @param params Vector of Parameter pointers to update.
         */
        void step(std::vector<Parameter*>& params) override;

        /**
         * @brief Factory method creating a unique pointer to a Momentum optimizer.
         * @param rate Learning rate @f$\eta@f$.
         * @param beta Momentum coefficient @f$\beta@f$ (default: 0.5).
         * @return std::unique_ptr<Momentum> Newly created optimizer.
         */
        static std::unique_ptr<Momentum> create(cum::cumeric_t rate, cum::cumeric_t beta = 0.5);

    private:
        cum::cumeric_t b;                                      /**< Momentum decay factor beta. */
        std::unordered_map<Parameter*, cum::Tensor> momentum;  /**< Accumulated velocity buffers per parameter. */
    };
}
