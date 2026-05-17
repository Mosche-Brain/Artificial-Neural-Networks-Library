#pragma once

#include "Core.hpp"

#include "cum/Vector.hpp"

namespace cum
{
    class Vector;
    enum MemoryAlingment { RowMajor, ColMajor };
    class Matrix
    {
    public:
        Matrix(std::size_t rows, std::size_t cols, cumeric_t value=0);
        Matrix() = default;
        ~Matrix();

        // operator Vector() const;

        cumeric_t& at(const std::size_t row, const std::size_t col) { return data_[get_idx(row, col)]; }
        const cumeric_t& at(const std::size_t row, const std::size_t col) const { return data_[get_idx(row, col)]; }
        std::size_t rows() const { return rows_; }
        std::size_t cols() const { return cols_; }

        Matrix row(const std::size_t i) const;
        Matrix col(const std::size_t i) const;

        std::size_t size() const { return rows_ * cols_; }

        cumeric_t& operator () (const std::size_t row, const std::size_t col) { return data_[get_idx(row, col)]; };
        const cumeric_t& operator () (const std::size_t row, const std::size_t col) const { return data_[get_idx(row, col)]; };

        Matrix& operator = (const Matrix& other);

        Matrix& operator += (const Matrix& other);
        Matrix& operator -= (const Matrix& other);
        Matrix& operator *= (const Matrix& other);
        Matrix& operator *= (const cumeric_t& scalar);
        Matrix& operator /= (const Matrix& other);
        Matrix& operator /= (const cumeric_t& scalar);
  

        friend Matrix operator + (const Matrix& A, const Matrix& B);
        friend Matrix operator - (const Matrix& A, const Matrix& B);
        friend Matrix operator * (const Matrix& A, const Matrix& B);
        // friend Matrix operator * (const Matrix& A, const Vector& v);
        friend Matrix operator * (const Matrix& mat, const cumeric_t& scalar);
        friend Matrix operator / (const Matrix& A, const Matrix& B);
        friend Matrix operator / (const Matrix& mat, const cumeric_t& scalar);

        friend Matrix activation(const Matrix& mat, const char* name);
        friend Matrix activationInPlace(Matrix& mat, const char* name);

        cumeric_t* data() { return data_; };

        Matrix transpose();
        Matrix& transposeInPlace();
        Matrix cwiseProduct(const Matrix& other);
        Matrix& cwiseProductInPlace();

        Matrix transform(void (*func)(cumeric_t* data, const std::size_t size)) const;
        Matrix& transformInPlace(void (*func)(cumeric_t* data, const std::size_t size));

        Matrix transform(cumeric_t (*func)(cumeric_t x)) const;
        Matrix& transformInPlace(cumeric_t (*func)(cumeric_t x));

        Matrix& rowwiseOpInPlace(void (*op)(cumeric_t* row, const cumeric_t* v, const std::size_t cols), const cumeric_t* arr);
        Matrix& colwiseOpInPlace(void (*op)(cumeric_t* col, const cumeric_t* v, const std::size_t rows), const cumeric_t* arr);

        cumeric_t norm();
        cumeric_t squaredNorm();

        Matrix colwiseSum();
        Vector colwiseSum2vec();
        Matrix rowwiseSum();
        Vector rowwiseSum2vec();

        Vector flatten() const;
    private:
        std::size_t get_idx(const std::size_t row, const std::size_t col) const { return row * cols_ + col; }

        std::size_t rows_, cols_;
        cumeric_t* data_;
    };
} // namespace cum
