//
// Created by jaro on 7/19/26.
//

#pragma once

#include "cum/Core.hpp"

#include <cstddef>

namespace cum::functions::trigonometric
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t sin(cumeric_t x);
    cumeric_t cos(cumeric_t x);
    cumeric_t tan(cumeric_t x);

    /* Parallel versions */

    __event__ sin(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ sin_in_place(cumeric_t* v, std::size_t N);

    __event__ cos(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ cos_in_place(cumeric_t* v, const std::size_t N);

    __event__ tan(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ tan_in_place(cumeric_t* v, std::size_t N);

    /* ========================== Derivatives ========================== */

    /* Scalar versions */

    cumeric_t sin_deriv(cumeric_t x);
    cumeric_t cos_deriv(cumeric_t x);
    cumeric_t tan_deriv(cumeric_t x);
    cumeric_t tan_deriv_from_result(cumeric_t x);

    /* Parallel versions */

    __event__ sin_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ sin_deriv_in_place(cumeric_t* v, std::size_t N);

    __event__ cos_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ cos_deriv_in_place(cumeric_t* v, std::size_t N);

    __event__ tan_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ tan_deriv_in_place(cumeric_t* v, std::size_t N);

    __event__ tan_deriv_from_result(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ tan_deriv_in_place_from_result(cumeric_t* v, std::size_t N);

} // namespace cum::functions::trigonometric