//
// Created by jaro on 7/19/26.
//

#pragma once

#include "cum/Core.hpp"

#include <cstddef>

// TODO: Dodać przeładowania in_place

namespace cum::functions::exponential
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t exp(cumeric_t x);
    cumeric_t exp2(cumeric_t x);
    cumeric_t exp10(cumeric_t x);

    cumeric_t log(cumeric_t x);   // base e
    cumeric_t log2(cumeric_t x);
    cumeric_t log10(cumeric_t x);

    /* Parallel versions */

    __event__ exp(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ exp_in_place(cumeric_t* v, std::size_t N);
    __event__ exp2(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ exp10(cumeric_t* r, const cumeric_t* v, std::size_t N);

    __event__ log(cumeric_t* r, const cumeric_t* v, std::size_t N);   // base e
    __event__ log2(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ log10(cumeric_t* r, const cumeric_t* v, std::size_t N);

    /* ========================== Derivatives ========================== */

    /* Scalar versions */

    cumeric_t exp_deriv(cumeric_t x);
    cumeric_t exp_deriv_from_result(cumeric_t x);

    cumeric_t exp2_deriv(cumeric_t x);
    cumeric_t exp2_deriv_from_result(cumeric_t x);

    cumeric_t exp10_deriv(cumeric_t x);
    cumeric_t exp10_deriv_from_result(cumeric_t x);

    cumeric_t log_deriv(cumeric_t x);
    cumeric_t log2_deriv(cumeric_t x);
    cumeric_t log10_deriv(cumeric_t x);

    /* Parallel versions */

    __event__ exp_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ exp_deriv_in_place(cumeric_t* v, std::size_t N);

    __event__ exp2_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ exp2_deriv_in_place(cumeric_t* v, std::size_t N);

    __event__ exp10_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ exp10_deriv_in_place(cumeric_t* v, std::size_t N);

    __event__ log_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ log_deriv_in_place(cumeric_t* v, std::size_t N);

    __event__ log2_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ log2_deriv_in_place(cumeric_t* v, std::size_t N);

    __event__ log10_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    __event__ log10_deriv_in_place(cumeric_t* v, std::size_t N);

} // namespace cum::functions::exponential
