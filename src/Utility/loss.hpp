#pragma once 

#include "math_api.hpp"
#include <stdexcept>
#include <limits>
namespace YANN::Utils::loss
{
    typedef struct { numeric_t loss ; matrix_t gradient; } LossType;
    
    enum class LossFunction { mse, cross_entropy, binary_cross_entropy }     ;

    LossType computeLoss(const matrix_t& result, const matrix_t& target, LossFunction loss_function);

    // matrix_t mse(matrix_t, matrix_t target);
    // f_type mse(matrix_t, matrix_t target);
    LossType mse(const matrix_t& result, const matrix_t& target);
    // matrix_t cross_entropy(matrix_t result, matrix_t target);
    LossType binary_cross_entropy(const matrix_t& result, const matrix_t& target);
    LossType cross_entropy(const matrix_t& result, const matrix_t& target);
}