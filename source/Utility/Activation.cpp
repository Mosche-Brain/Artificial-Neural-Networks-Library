#include "Activation.hpp"

namespace SNN::Utils
{
    Activation stringToFunction(const char* func)
    {
        std::map<const char*, Activation> map = 
        {
            {"tanh", {tanhf, } }
        };


    }
}