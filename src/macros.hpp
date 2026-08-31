//
// Created by jaro on 8/31/26.
//

#ifndef YANN_MACROS_HPP
#define YANN_MACROS_HPP

#include <cum/Core.hpp>

namespace cum
{
    class Tensor;
    class Matrix;
    class Vector;
}

namespace yann
{
    using scalar_t = cum::cumeric_t;
    using Vector = cum::Vector;
    using Matrix = cum::Matrix;
    using Tensor = cum::Tensor;
}

#endif //YANN_MACROS_HPP
