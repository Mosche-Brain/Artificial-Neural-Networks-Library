//
// Created by jaro on 7/19/26.
//

#pragma once
#include "cum/Core.hpp"

namespace cum::functions::cyclometric
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t asin(cumeric_t x);
    cumeric_t acos(cumeric_t x);
    cumeric_t atan(cumeric_t x);

    /* Parallel versions */

    void asin(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void asin_in_place(cumeric_t* v, std::size_t N);

    void acos(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void acos_in_place(cumeric_t* v, std::size_t N);

    void atan(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void atan_in_place(cumeric_t* v, std::size_t N);

    /* ========================== Derivatives ========================== */

    /* Scalar versions */

    cumeric_t asin_deriv(cumeric_t x);
    cumeric_t acos_deriv(cumeric_t x);
    cumeric_t atan_deriv(cumeric_t x);

    /* Parallel versions */

    void asin_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void asin_deriv_in_place(cumeric_t* v, std::size_t N);

    void acos_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void acos_deriv_in_place(cumeric_t* v, std::size_t N);

    void atan_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void atan_deriv_in_place(cumeric_t* v, std::size_t N);

}
