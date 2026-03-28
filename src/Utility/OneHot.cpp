#include "OneHot.hpp"

namespace YANN::Utils
{
    size_t OneHotDecode(const vector_t& vector)
    {
        for(size_t i = 0 ; i < vector.size() ; i++)
        {
            if(vector[i] == 1.0f)
            {
                return i;
            }
        }
    }

    vector_t OneHotEncode(size_t label, size_t lenght)
    {
        vector_t vector = vector_t::Zero(lenght);
        vector[label] = 1.0f;

        return vector;
    }
}