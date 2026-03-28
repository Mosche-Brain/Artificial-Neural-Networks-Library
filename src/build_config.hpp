#pragma once

#include <eigen3/Eigen/Dense>
#include <cmath>
#include <cstdint>

typedef _Float16 float_t;

typedef Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
typedef Eigen::Matrix<float_t, Eigen::Dynamic, 1> vector_t;