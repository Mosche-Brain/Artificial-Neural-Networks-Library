//
// Created by jaro on 7/19/26.
//

#pragma once

#include "cum/Core.hpp"

namespace cum::functions::various
{
    void fill(cumeric_t* v, cumeric_t val, std::size_t N);

    cumeric_t clamp(cumeric_t a, cumeric_t min, cumeric_t max);
    // cummulative_t clamp(cummulative_t a, cummulative_t min, cummulative_t max);

    void clip(cumeric_t* r, const cumeric_t* v, cumeric_t min, cumeric_t max, std::size_t N);
    void clipInPlace(cumeric_t* v, cumeric_t min, cumeric_t max, std::size_t N);

    cumeric_t abs(cumeric_t x);

    cumeric_t min(cumeric_t a, cumeric_t b);
    cumeric_t max(cumeric_t a, cumeric_t b);
}