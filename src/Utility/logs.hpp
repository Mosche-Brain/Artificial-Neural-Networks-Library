#pragma once

#include <Eigen/Dense>

#include <iostream>

namespace ANN::Utils::logs
{
    std::string show_matrix_dimensions(const Eigen::MatrixXf& mat);
}