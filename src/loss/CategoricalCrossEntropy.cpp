/*
 * @author: jaro
 * @name:   CategoricalCrossEntropy
 * @file:   src/loss/CategoricalCrossEntropy.cpp
 * @date:   29 September 2026 18:50:55
 */

#include "yann/loss/CategoricalCrossEntropy.hpp"

namespace yann::loss
{
    void loss::CategoricalCrossEntropy::compute(const cum::Tensor& predicted, const cum::Tensor& target)
    {

    }

    std::unique_ptr<LossBase> CategoricalCrossEntropy::create()
    {
        return std::make_unique<CategoricalCrossEntropy>();
    }

} // yann