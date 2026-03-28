#pragma once

#include "build_config.hpp"
#include <Eigen/Dense>
#include <iostream>

namespace YANN::Utils::logs
{
    std::string show_matrix_dimensions(const matrix_t& mat);
}