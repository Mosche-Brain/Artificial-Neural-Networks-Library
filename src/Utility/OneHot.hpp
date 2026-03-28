#pragma once

#include <Eigen/Dense>

#include <build_config.hpp>

namespace YANN::Utils
{
    size_t OneHotDecode(const vector_t& vector);
    vector_t OneHotEncode(size_t label, size_t lengt);
}