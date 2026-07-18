#pragma once

#include "cum/Core.hpp"

namespace cum::random
{
    void uniform(cumeric_t* buff, const std::size_t N, const cumeric_t min=-1._c, const cumeric_t max=1._c);
    void normal(cumeric_t* buff, const std::size_t N, const cumeric_t sigma, std::size_t seed=2137);
}