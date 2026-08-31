/* Created by jaro on 7/17/26. */

#ifndef YANN_OPTIMIZERBASE_HPP
#define YANN_OPTIMIZERBASE_HPP

#include <vector>

#include "Parameter.hpp"

namespace cum { class Matrix; }

namespace yann::optimizers
{
    class OptimizerBase
    {
    public:
        explicit OptimizerBase(cum::cumeric_t learning_rate) : learning_rate(learning_rate) {}
        virtual ~OptimizerBase() = default;
        virtual void step(cum::Matrix& params, cum::Matrix& grad) = 0;
        virtual void step(Parameter& param) {};
        virtual void step(std::vector<Parameter*>& params) = 0;

        virtual void scale_grads(std::vector<Parameter*>& params, cum::cumeric_t scalar);
        virtual void clear_grads(std::vector<Parameter*>& params);

        cum::cumeric_t learning_rate;
    };

    using Optimizer = std::unique_ptr<OptimizerBase>;
} // yann

#endif //YANN_OPTIMIZERBASE_HPP
