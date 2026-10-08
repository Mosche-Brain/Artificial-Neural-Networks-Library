//
// Created by jaro on 7/17/26.
//

#include <print>

#include <cum/runtime.hpp>

#include "yann/logging/Logger.hpp"

#include "yann/optimizers/SGD.hpp"

namespace yann::optimizers
{
    SGD::SGD(cum::cumeric_t learning_rate)
        : OptimizerBase(learning_rate)
    {
    }

    void SGD::step(cum::Tensor& params, cum::Tensor& grad)
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
            YANN_LOG(2, "values norm: {} | grad norm: {}", param->values.norm(), param->gradient.norm());
            param->values -= param->gradient * learning_rate;
            param->clear_gradient();
        }
    }

    std::unique_ptr<SGD> SGD::create(cum::cumeric_t learning_rate)
    {
        return std::make_unique<SGD>(learning_rate);
    }
} // namespace yann::optimizers
