#pragma once

#include <vector>

struct ResponseCurve
{
    std::vector<float> xs;
    std::vector<float> ys;
    std::vector<float> trainXs;
    std::vector<float> trainYs;
    float markerX = 0.0f;
    float markerY = 0.0f;
    bool valid = false;
};
