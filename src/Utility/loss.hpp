#pragma once 

#include <Eigen/Dense>

#include <stdexcept>
#include <limits>
namespace ANN::Utils::loss
{
    typedef struct { float_t loss ; Eigen::MatrixXf gradient; } LossType;
    
    enum class LossFunction { mse, cross_entropy, binary_cross_entropy }     ;

    LossType computeLoss(const Eigen::MatrixXf& result, const Eigen::MatrixXf& target, LossFunction loss_function);

    // Eigen::MatrixXf mse(Eigen::MatrixXf, Eigen::MatrixXf target);
    // float_t mse(Eigen::MatrixXf, Eigen::MatrixXf target);
    LossType mse(const Eigen::MatrixXf& result, const Eigen::MatrixXf& target);
    // Eigen::MatrixXf cross_entropy(Eigen::MatrixXf result, Eigen::MatrixXf target);
    LossType binary_cross_entropy(const Eigen::MatrixXf& result, const Eigen::MatrixXf& target);
    LossType cross_entropy(const Eigen::MatrixXf& result, const Eigen::MatrixXf& target);
}