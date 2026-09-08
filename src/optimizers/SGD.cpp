//
// Created by jaro on 7/17/26.
//

#include "optimizers/SGD.hpp"

#include "cum/runtime.hpp"

namespace yann::optimizers
{
    SGD::SGD(cum::cumeric_t learning_rate)
        : OptimizerBase(learning_rate)
    {
    }

    void SGD::step(cum::Matrix& params, cum::Matrix& grad)
    {
    }

    void SGD::step(Parameter& param)
    {
        param.values -= param.gradient * learning_rate;
        param.clear_gradient();
    }

    void SGD::step(std::vector<Parameter*>& params)
    {
        for (Parameter* param : params)
        {
            param->values -= param->gradient * learning_rate;
            param->clear_gradient();
        }
    }

    std::unique_ptr<SGD> SGD::create(cum::cumeric_t learning_rate)
    {
        return std::make_unique<SGD>(learning_rate);
    }
} // namespace yann::optimizers
