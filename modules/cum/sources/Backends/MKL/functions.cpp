#include "cum/functions.hpp"

#include <math.h>
#include <string.h>

namespace cum::functions
{
    cumeric_t relu(cumeric_t x) 
    {
        return x > 0 ? x : 0;
    }
    cumeric_t ReLU::operator()(cumeric_t x) const
    {
        return relu(x);
    }

    cumeric_t Tanh::operator()(cumeric_t x) const
    {
        return tanhf(x);
    }

    void (*getFunctionByName(const char* name))(cumeric_t)
    {
        if(strcmp(name, "relu") == 0)
            return relu;
        else if(strcmp(name, "tanh") == 0)
            return Tanh();
        else
            return nullptr;
    }
    

} // namespace cum::functions
