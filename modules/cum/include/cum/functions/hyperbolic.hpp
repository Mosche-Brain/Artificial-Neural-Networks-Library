//
// Created by jaro on 7/19/26.
//

#ifndef CUM_FUNCTIONS_HYPERBOLIC_HPP
#define CUM_FUNCTIONS_HYPERBOLIC_HPP

#include "cum/Core.hpp"

#include <cstddef>

namespace cum::functions::hyperbolic
{
    /* ========================== Base Functions ========================== */

    /* Scalar versions */

    cumeric_t tanh(cumeric_t x);
    cumeric_t sinh(cumeric_t x);
    cumeric_t cosh(cumeric_t x);

    /* Parallel versions */

    void tanh(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void tanh_in_place(cumeric_t* v, std::size_t N);

    void sinh(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void sinh_in_place(cumeric_t* v, std::size_t N);

    void cosh(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void cosh_in_place(cumeric_t* v, std::size_t N);

    /* ========================== Derivatives ========================== */

    /* Scalar versions */

    cumeric_t tanh_deriv(cumeric_t x);
    cumeric_t tanh_deriv_from_result(cumeric_t x);
    cumeric_t sinh_deriv(cumeric_t x);
    cumeric_t cosh_deriv(cumeric_t x);

    /* Parallel versions */

    void tanh_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void tanh_deriv_from_result(cumeric_t* r, const cumeric_t* v, std::size_t N);

    void sinh_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void sinh_deriv_in_place(cumeric_t* v, std::size_t N);

    void cosh_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N);
    void cosh_deriv_in_place(cumeric_t* v, std::size_t N);

} // namespace cum::functions::hyperbolic

#endif // CUM_FUNCTIONS_HYPERBOLIC_HPP