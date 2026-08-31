//
// Created by jaro on 7/17/26.
//

#include "yann/optimizers/OptimizerBase.hpp"

namespace yann::optimizers
{
    void OptimizerBase::scale_grads(std::vector<Parameter*>& params, const cum::cumeric_t scalar)
    {
        for (auto param : params)
            param->scale_gradient(scalar);
    }

    void OptimizerBase::clear_grads(std::vector<Parameter*>& params)
    {
        for (auto param : params)
            param->clear_gradient();
    }
} // yann