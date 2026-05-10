#pragma once

#include "Core.hpp"

namespace cum
{
    enum MemoryAlingment { RowMajor, ColMajor };
    class Matrix
    {
    public:
        Matrix(std::size_t rows, std::size_t cols, cumeric_t value=0);
        Matrix() = default;
        ~Matrix();

        cumeric_t& at(std::size_t row, std::size_t col);

        std::size_t rows() { return rows_; }
        std::size_t cols() { return cols_; }

        Matrix& operator += (const Matrix& other);
        Matrix& operator -= (const Matrix& other);
        Matrix& operator *= (const Matrix& other);
        friend Matrix operator + (const Matrix& A, const Matrix& B);
        friend Matrix operator * (const Matrix& A, const Matrix& B);

        void transpose();
        void cwiseProduct();
    private:
        std::size_t index(std::size_t row, std::size_t col);

        std::size_t rows_, cols_;
        cumeric_t* data;
    };
} // namespace cum
