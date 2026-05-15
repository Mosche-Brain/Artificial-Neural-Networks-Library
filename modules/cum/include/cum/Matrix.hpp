#pragma once

#include "Core.hpp"

#include "cum/Vector.hpp"

namespace cum
{
    enum MemoryAlingment { RowMajor, ColMajor };
    class Matrix
    {
    public:
        Matrix(std::size_t rows, std::size_t cols, cumeric_t value=0);
        Matrix() = default;
        ~Matrix();

        cumeric_t& at(const std::size_t row, const std::size_t col) { return data_[get_idx(row, col)]; }

        std::size_t rows() { return rows_; }
        std::size_t cols() { return cols_; }

        Matrix row(std::size_t i) const;
        Matrix col(std::size_t i) const;

        std::size_t size() const { return rows_ * cols_; }

        cumeric_t& operator () (const std::size_t row, const std::size_t col) { return data_[get_idx(row, col)]; };

        Matrix& operator = (const Matrix& other);

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
        Matrix& transposeInPlace();
        Matrix cwiseProduct(const Matrix& other);
        Matrix& cwiseProductInPlace();

        Matrix transform(void (*func)(cumeric_t* data, const std::size_t size)) const;
        Matrix& transformInPlace(void (*func)(cumeric_t* data, const std::size_t size));

        Matrix& rowwiseOpInPlace(void (*op)(cumeric_t* row, const cumeric_t* v, const std::size_t cols), const cumeric_t* arr);
        Matrix& colwiseOpInPlace(void (*op)(cumeric_t* col, const cumeric_t* v, const std::size_t rows), const cumeric_t* arr);

        Matrix colwiseSum();
        Matrix rowwiseSum();
    private:
        std::size_t get_idx(const std::size_t row, const std::size_t col) { return row * cols_ + col; }

        std::size_t rows_, cols_;
        cumeric_t* data_;
    };
} // namespace cum
