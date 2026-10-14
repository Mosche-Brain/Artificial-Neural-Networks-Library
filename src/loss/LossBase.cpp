//
// Created by jaro on 7/31/26.
//

#include "yann/loss/LossBase.hpp"

namespace yann::loss
{
    void LossBase::compute(const cum::Tensor& predicted, const cum::Tensor& target, Reduction reduction)
    {
        if (predicted.shape() != target.shape())
            throw std::invalid_argument("Loss::compute: predicted and target shapes must match");
        if (predicted.shape()[batch_axis] == 0)
            throw std::invalid_argument("Loss::compute: batch is empty");

        compute_sum(predicted, target);

        if (reduction == Reduction::BatchMean)
        {
            const cum::cumeric_t batch = static_cast<cum::cumeric_t>(predicted.shape()[batch_axis]);
            loss.value /= batch;
            loss.gradient = loss.gradient / batch;
        }
    }
} // yann