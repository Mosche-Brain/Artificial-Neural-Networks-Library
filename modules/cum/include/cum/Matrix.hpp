#pragma once

#include "Core.hpp"

#include "cum/Vector.hpp"

#include <initializer_list>

namespace cum
{
    class Vector;
    enum MemoryAlingment { RowMajor, ColMajor };
    class Matrix
    {
    public:
        Matrix(std::size_t rows, std::size_t cols, cumeric_t value=0);
        Matrix(std::size_t rows, std::size_t cols, cumeric_t* source); // this constructor is used for copying data from source, not set source as a data handle
        Matrix(std::size_t rows, std::size_t cols, std::initializer_list<cumeric_t> elements);
        Matrix(const Matrix& other);
        Matrix(Matrix&& other) noexcept;
        Matrix() = default;
        ~Matrix();

        /* functions for advanced initialization */
        static Matrix Random(std::size_t rows, std::size_t cols, cumeric_t min = -1_c, cumeric_t max = 1_c);
        static Matrix Zeros(std::size_t rows, std::size_t cols);
        static Matrix Ones(std::size_t rows, std::size_t cols);
        static Matrix Linspace(cumeric_t start, cumeric_t end, std::size_t num); /* Row Vector */
        // operator Vector() const;

        void fill(cumeric_t value);

        std::size_t rows() const { return rows_; }
        std::size_t cols() const { return cols_; }
        
        Matrix row(const std::size_t i) const;
        Matrix col(const std::size_t i) const;
        
        std::size_t size() const { return rows_ * cols_; }
        
        /* Accesors */
        cumeric_t* data() { return data_; };
        const cumeric_t* data() const { return data_; };

        cumeric_t& at(const std::size_t row, const std::size_t col) { return data_[get_idx(row, col)]; }
        const cumeric_t& at(const std::size_t row, const std::size_t col) const { return data_[get_idx(row, col)]; }

        cumeric_t& operator () (const std::size_t row, const std::size_t col) { return data_[get_idx(row, col)]; };
        const cumeric_t& operator () (const std::size_t row, const std::size_t col) const { return data_[get_idx(row, col)]; };

        /* assingment operator */
        // Matrix& operator = (const Matrix& other);
        // Matrix& operator = (const Matrix& other);
        Matrix& set(const Matrix& other);
        Matrix& operator = (Matrix other) noexcept;

        void swap(Matrix& other);
        friend void swap(Matrix& A, Matrix& B);

        /* Inplace arithmetic operator */
        Matrix& operator += (const Matrix& other);
        Matrix& operator -= (const Matrix& other);
        Matrix& operator *= (const Matrix& other);
        Matrix& operator *= (const cumeric_t& scalar);
        Matrix& operator /= (const Matrix& other);
        Matrix& operator /= (const cumeric_t& scalar);
  
        /* Outplace arithmetic operator */
        friend Matrix operator + (const Matrix& A, const Matrix& B);
        friend Matrix operator - (const Matrix& A, const Matrix& B);
        friend Matrix operator * (const Matrix& A, const Matrix& B);
        // friend Matrix operator * (const Matrix& A, const Vector& v);
        friend Matrix operator * (const Matrix& mat, const cumeric_t& scalar);
        friend Matrix operator / (const Matrix& A, const Matrix& B);
        friend Matrix operator / (const Matrix& mat, const cumeric_t& scalar);

        
        /* Transforming matrix dimensions */
        Matrix transpose();
        Matrix& transposeInPlace();
        Vector flatten() const;
        Matrix reshape(const std::size_t rows, const std::size_t cols) const;
        
        friend Matrix activation(const Matrix& mat, const char* name);
        friend Matrix activationInPlace(Matrix& mat, const char* name);

        Matrix cwiseProduct(const Matrix& other);
        Matrix& cwiseProductInPlace();

        Matrix transform(void (*func)(cumeric_t* data, const std::size_t size)) const;
        Matrix& transformInPlace(void (*func)(cumeric_t* data, const std::size_t size));

        Matrix transform(cumeric_t (*func)(cumeric_t x)) const;
        Matrix& transformInPlace(cumeric_t (*func)(cumeric_t x));

        Matrix& rowwiseOpInPlace(void (*op)(cumeric_t* row, const cumeric_t* v, const std::size_t cols), const cumeric_t* arr);
        Matrix& colwiseOpInPlace(void (*op)(cumeric_t* col, const cumeric_t* v, const std::size_t rows), const cumeric_t* arr);

        Matrix clip(const cumeric_t min, const cumeric_t max) const;
        Matrix& clipInPlace(const cumeric_t min, const cumeric_t max);

        cumeric_t norm();
        cumeric_t squaredNorm();

        Matrix colwiseSum();
        Vector colwiseSum2vec();
        Matrix rowwiseSum();
        Vector rowwiseSum2vec();

        Matrix shuffleRows() const;
        Matrix& shuffleRowsInPlace();

    private:
        std::size_t get_idx(const std::size_t row, const std::size_t col) const { return row * cols_ + col; }

        std::size_t rows_, cols_;
        cumeric_t* data_=nullptr;
    };

} // namespace cum
