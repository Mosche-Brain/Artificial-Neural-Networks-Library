#pragma once

#include <stdfloat>

#include "cum/Core.hpp"
#include "cum/Matrix.hpp"
#include "cum/LinearAlgebra.hpp"

namespace cum
{
    void cum(CUM_DEVICE device = CUM_DEVICE::AUTO);
    void recum(CUM_DEVICE device = CUM_DEVICE::AUTO);
    void decum();
}