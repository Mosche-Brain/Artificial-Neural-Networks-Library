#pragma once

#include "Core.hpp"

namespace cum
{
    enum MemoryAlingment { RowMajor, ColMajor };
    class Matrix
    {
    public:
        Matrix(std::size_t rows, std::size_t cols, cumeric_t value);

        cumeric_t at(std::size_t row, std::size_t col);

    private:
        std::size_t rows_, cols_;
        cumeric_t* data;
    };
} // namespace cum
