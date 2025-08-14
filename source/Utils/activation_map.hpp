#pragma once

#include <map>

#include "Utils/functions.hpp"

struct activation_t
{
    std::function<double(double)> function;
    std::function<double(double)> derivative;
};
 
std::map<const char*, activation_t> activation_map =
{
   {"linear",   {pass, pass_prim}},
   {"relu",     {ReLU, ReLU_prim}},
   {"gelu",     {GELU, GELU_prim}},
   {"tanh",     {tanh, tanh_prim}},
   {"sigmoid",  {sigmoid, sigmoid_prim}}
};