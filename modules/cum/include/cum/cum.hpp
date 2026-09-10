#pragma once

#include <stdfloat>

#include "cum/Core.hpp"
#include "cum/Matrix.hpp"
#include "cum/LinearAlgebra.hpp"

namespace cum
{
    void cum(DEVICE device = DEVICE::AUTO);
    void recum(DEVICE device = DEVICE::AUTO);
    void decum();
}