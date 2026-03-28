#pragma once

#include <eigen3/Eigen/Dense>
#include <cstdint>

namespace YANN
{
    // typedef _Float16 numeric_t;
    typedef _Float16 numeric_t;

    typedef Eigen::Matrix<numeric_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
    typedef Eigen::Matrix<numeric_t, Eigen::Dynamic, 1> vector_t;
}