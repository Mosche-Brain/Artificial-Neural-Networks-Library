#pragma once

#include "math_api.hpp"

#include "Sequential.hpp"

namespace YANN::Utils::FileIO
{
    void saveMatrix(YANN::matrix_t  mat, const char* filename);
    void readMatrix(YANN::matrix_t& mat, const char* filename);

    void saveSequentialModel(const Models::Sequential& model, const char* filename);
    void loadSequentialModel(Models::Sequential& model, const char* filename);
}