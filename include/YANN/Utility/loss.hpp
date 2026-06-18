#pragma once 
#include <stdexcept>
#include <limits>

#include <cum/Matrix.hpp>

namespace YANN::Utils::loss
{
    typedef struct { cum::cummulative_t loss ; cum::Matrix gradient; } LossType;
    
    enum class LossFunction { mse, cross_entropy, binary_cross_entropy };

    LossType computeLoss(const cum::Matrix& result, const cum::Matrix& target, LossFunction loss_function);

    LossType mse(const cum::Matrix& result, const cum::Matrix& target);
    LossType binary_cross_entropy(const cum::Matrix& result, const cum::Matrix& target);
    LossType cross_entropy(const cum::Matrix& result, const cum::Matrix& target);
}