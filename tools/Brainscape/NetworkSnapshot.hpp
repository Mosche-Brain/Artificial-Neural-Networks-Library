#pragma once

#include <vector>

struct NetworkSnapshot
{
    std::vector<int> layerSizes;
    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<float>> outputs;
    float x = 0.0f;
    float yHat = 0.0f;
    bool valid = false;
};
