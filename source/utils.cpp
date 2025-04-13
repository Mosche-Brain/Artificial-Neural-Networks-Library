#include "utils.hpp"

double random(double min, double max) 
{
    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen); 
}

std::vector<double> randomVector(size_t n, double min, double max)
{
    std::vector<double> temp;
    temp.reserve(n);

    for(size_t i = 0 ; i < n ; i++)
    {
        temp.push_back(random(min, max));
    }

    return temp;
}