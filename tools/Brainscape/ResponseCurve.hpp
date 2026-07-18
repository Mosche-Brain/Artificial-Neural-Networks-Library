#pragma once

#include <vector>

struct ResponseCurve
{
    std::vector<float> xs;
    std::vector<float> ys;
    std::vector<float> trainXs;
    std::vector<float> trainYs;
    std::vector<float> trainPredYs;
    float markerX = 0.0f;
    float markerY = 0.0f;
    float mse = 0.0f;
    bool valid = false;
};
