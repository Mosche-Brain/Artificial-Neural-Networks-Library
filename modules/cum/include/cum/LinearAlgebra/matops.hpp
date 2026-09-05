// Created by Jaro 8/30/26
#pragma once
#include "cum/Core.hpp"
#include "BLAS/level3.hpp"

namespace cum::LinearAlgebra
{
	__event__ mmul(cumeric_t* C, const cumeric_t* A, const cumeric_t*, dim_t m, dim_t n, dim_t k, __depencies__ depencies={});

	__event__ mtrans(cumeric_t* At, const cumeric_t* A, dim_t m, dim_t n, __depencies__ depencies={});
	__event__ mtrans(cumeric_t* A, dim_t m, dim_t n, __depencies__ depencies={});
}
