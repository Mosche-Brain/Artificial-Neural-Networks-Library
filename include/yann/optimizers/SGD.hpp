/**
 * @file SGD.hpp
 * @author jaro
 * @brief Stochastic Gradient Descent (SGD) optimizer implementation.
 * @date 2026-07-17
 */

#ifndef YANN_SGD_HPP
#define YANN_SGD_HPP

#include "optimizers/OptimizerBase.hpp"

#include <cum/cum.hpp>
#include <vector>
#include <memory>

namespace yann::optimizers
{
    /**
     * @class SGD
     * @brief Stochastic Gradient Descent optimizer updating parameters via:
     * @f[ \theta \leftarrow \theta - \eta \cdot \nabla_\theta L @f]
     */
    class SGD : public OptimizerBase
    {
    public:
        /**
         * @brief Construct an SGD optimizer with a specified learning rate.
         * @param learning_rate Learning rate step size @f$\eta@f$.
         */
        SGD(cum::cumeric_t learning_rate);

        /**
         * @brief Update a single Parameter instance.
         * @param param Parameter instance holding values and gradient.
         */
        void step(Parameter& param) override;

        /**
         * @brief Update multiple Parameter instances in a single optimization step.
         * @param params Vector of Parameter pointers to update.
         */
        void step(std::vector<Parameter*>& params) override;

        /**
         * @brief Factory method creating a unique pointer to an SGD optimizer.
         * @param learning_rate Learning rate step size.
         * @return std::unique_ptr<SGD> Newly created optimizer.
         */
        static std::unique_ptr<SGD> create(cum::cumeric_t learning_rate);

    private:
        // std::vector<cum::Matrix*> weights_grad;
        // std::vector<cum::Matrix*> biases_grad; // column vector
        // std::vector<cum::Matrix*> weights_;
        // std::vector<cum::Matrix*> biases_; // column vector
    };
} // yann

#endif //YANN_SGD_HPP
