#include "OneHot.hpp"

namespace YANN::Utils
{
    size_t OneHotDecode(const Eigen::VectorXf& vector)
    {
        for(size_t i = 0 ; i < vector.size() ; i++)
        {
            if(vector[i] == 1.0f)
            {
                return i;
            }
        }
    }

    Eigen::VectorXf OneHotEncode(size_t label, size_t lenght)
    {
        Eigen::VectorXf vector = Eigen::VectorXf::Zero(lenght);
        vector[label] = 1.0f;

        return vector;
    }
}