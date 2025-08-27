#pragma once

#include <Eigen/Dense>

namespace SNN::Utils
{
    size_t OneHotDecode(const Eigen::VectorXf& vector);
    Eigen::VectorXf OneHotEncode(size_t label, size_t lengt);
}