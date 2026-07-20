//
// Created by jaro on 7/19/26.
//

#pragma once

#include "cum/Core.hpp"

namespace cum::functions::various
{
    void fill(cumeric_t* v, cumeric_t val, const std::size_t N);

    void clip(cumeric_t* r, const cumeric_t* v, const cumeric_t min, const cumeric_t max, const std::size_t N);
    void clipInPlace(cumeric_t* v, const cumeric_t min, const cumeric_t max, const std::size_t N);

    cumeric_t min(const cumeric_t a, const cumeric_t b);
    cumeric_t max(const cumeric_t a, const cumeric_t b);
}