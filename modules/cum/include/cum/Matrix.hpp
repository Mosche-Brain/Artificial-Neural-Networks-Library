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

        cumeric_t& at(const std::size_t row, const std::size_t col);

        std::size_t rows() { return rows_; }
        std::size_t cols() { return cols_; }

        cumeric_t& operator () (const std::size_t row, const std::size_t col) { return data_[index(row, col)]; };

        Matrix& operator += (const Matrix& other);
        Matrix& operator -= (const Matrix& other);
        Matrix& operator *= (const Matrix& other);
        Matrix& operator /= (const Matrix& other);

        Matrix& operator *= (const cumeric_t& scalar);
        Matrix& operator /= (const cumeric_t& scalar);

        friend Matrix operator + (const Matrix& A, const Matrix& B);
        friend Matrix operator - (const Matrix& A, const Matrix& B);
        friend Matrix operator * (const Matrix& A, const Matrix& B);
        friend Matrix operator / (const Matrix& A, const Matrix& B);

        friend Matrix operator * (const Matrix& mat, const cumeric_t& scalar);
        friend Matrix operator / (const Matrix& mat, const cumeric_t& scalar);

        cumeric_t* data() { return data_; };

        Matrix transpose();
        void transposeInPlace();
        Matrix cwiseProduct(const Matrix& other);
        void cwiseProductInPlace();

        void rowwiseOpInPlace(void (*op)(cumeric_t* row, const cumeric_t* v, const std::size_t cols), const cumeric_t* arr);
        void colwiseOpInPlace(void (*op)(cumeric_t* col, const cumeric_t* v, const std::size_t rows), const cumeric_t* arr);
    private:
        std::size_t index(const std::size_t row, const std::size_t col);

        std::size_t rows_, cols_;
        cumeric_t* data_;
    };
} // namespace cum
