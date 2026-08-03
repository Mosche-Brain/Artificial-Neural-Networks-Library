//
// Created by jaro on 8/3/26.
//

#ifndef YANN_MATRIXVIEW_HPP
#define YANN_MATRIXVIEW_HPP

#include "Core.hpp"

#include "cum/Vector.hpp"

#include <initializer_list>

#include "cum/Matrix.hpp"

namespace cum
{
    class MatrixView
    {
    public:
        MatrixView(const Matrix& other);
        MatrixView(Matrix&& other) noexcept;

        MatrixView(cumeric_t* source, dim_t rows, dim_t cols);
        MatrixView(dim_t rows, dim_t cols, cumeric_t* source);
        MatrixView(const cumeric_t* source, dim_t rows, dim_t cols);
        MatrixView(dim_t rows, dim_t cols, const cumeric_t* source);

        MatrixView() = default;
        ~MatrixView();


        std::size_t rows() const { return rows_; }
        std::size_t cols() const { return cols_; }

        Matrix row(std::size_t i) const;
        Matrix col(std::size_t i) const;

        std::size_t size() const { return rows_ * cols_; }

        /* Accesors */
        const cumeric_t* data() const { return data_; };

        const cumeric_t& at(const std::size_t row, const std::size_t col) const { return data_[get_idx(row, col)]; }

        const cumeric_t& operator () (const std::size_t row, const std::size_t col) const { return data_[get_idx(row, col)]; };

        /* assingment operator */

        Matrix& set(const Matrix& other);
        // Matrix& operator = (Matrix other) noexcept;

        /* Outplace arithmetic operator */
        friend Matrix operator + (const Matrix& A, const Matrix& B);
        friend Matrix operator - (const Matrix& A, const Matrix& B);
        friend Matrix operator * (const Matrix& A, const Matrix& B);
        // friend Matrix operator * (const Matrix& A, const Vector& v);
        friend Matrix operator * (const Matrix& mat, const cumeric_t& scalar);
        friend Matrix operator / (const Matrix& A, const Matrix& B);
        friend Matrix operator / (const Matrix& mat, const cumeric_t& scalar);


        /* Transforming matrix dimensions */
        Matrix transpose() const;
        Vector flatten() const;
        Matrix reshape(std::size_t rows, std::size_t cols) const;

        friend Matrix activation(const Matrix& mat, const char* name);
        friend Matrix activationInPlace(Matrix& mat, const char* name);

        Matrix cwiseProduct(const Matrix& other);

        Matrix transform(void (*func)(cumeric_t* data, const std::size_t size)) const;

        Matrix transform(cumeric_t (*func)(cumeric_t x)) const;

        Matrix clip(cumeric_t min, cumeric_t max) const;

        cumeric_t norm();
        cumeric_t squaredNorm();

        Matrix colwiseSum();
        Vector colwiseSum2vec();
        Matrix rowwiseSum();
        Vector rowwiseSum2vec();

        Matrix shuffleRows() const;

    private:
        std::size_t get_idx(const std::size_t row, const std::size_t col) const { return row * cols_ + col; }

        std::size_t rows_, cols_;
        const cumeric_t* data_;
    };
} // cum

#endif //YANN_MATRIXVIEW_HPP
