/* Created by jaro on 7/20/26. */
#pragma once

#include "cum/Core.hpp"

#include "cum/functions/function_id.hpp"

namespace cum::functions
{
    void transform(cumeric_t* r, const cumeric_t* v, std::size_t N, function_id activation);
    void transform(cumeric_t* r, const cumeric_t* v, std::size_t N, const activation_t& activation);

    void transform_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N, function_id activation);
    void transform_deriv(cumeric_t* r, const cumeric_t* v, std::size_t N, const activation_t& activation);

}