#include "cum/functions.hpp"

namespace cum::functions
{
    cumeric_t ReLU::operator()(cumeric_t x) const
    {
        return x > 0 ? x : 0;
    }

    // cumeric_t Tanh::operator()(cumeric_t x) const
    // {
    //     return tanh;
    // }

    

} // namespace cum::functions
