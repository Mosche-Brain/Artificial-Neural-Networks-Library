#pragma once 
#include <stdexcept>
#include <limits>


#include "yann/loss/LossBase.hpp"
#include "yann/loss/MeanAbsoluteError.hpp"
#include "yann/loss/MeanSquaredError.hpp"
#include "yann/loss/MeanAbsoluteError.hpp"

/* --------------------------------------- */

namespace yann::loss
{
    typedef struct
    {
        cum::cummulative_t loss;
        cum::cummulative_t gradient_scale;
        cum::Matrix gradient;
    } LossType;

    // using loss_t = std::pair<cum::cummulative_t, cum::Matrix>;
    
    enum class LossFunction { mse, cross_entropy, binary_cross_entropy };

    LossType computeLoss(cum::Matrix& result, cum::Matrix& target, LossFunction loss_function);

    LossType mse(const cum::Matrix& result, const cum::Matrix& target);
    // LossType binary_cross_entropy(const cum::Matrix& result, const cum::Matrix& target);
    LossType binary_cross_entropy(cum::Matrix& result, cum::Matrix& target);
    LossType cross_entropy(const cum::Matrix& result, const cum::Matrix& target);
}