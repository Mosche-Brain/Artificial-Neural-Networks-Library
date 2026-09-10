#pragma once

#include "Core.hpp"

#include "cum/Vector.hpp"

#include <cstddef>
#include <initializer_list>

namespace cum
{
    class Vector;
    enum MemoryAlingment { RowMajor, ColMajor }; // not used yet
    class Matrix
    {
    public:
		Matrix(dim_t rows, dim_t cols);
        Matrix(dim_t rows, dim_t cols, cumeric_t value);
        Matrix(dim_t rows, dim_t cols, cumeric_t* source); // this constructor is used for copying data from source, not set source as a data handle_
        Matrix(dim_t rows, dim_t cols, const cumeric_t* source); // this constructor is used for copying data from source, not set source as a data handle_
        Matrix(dim_t rows, dim_t cols, std::initializer_list<cumeric_t> elements);
        Matrix(const Matrix& other);
        Matrix(Matrix&& other) noexcept;
        Matrix() = default;
        ~Matrix();

        /* functions for advanced initialization */
        static Matrix Random(dim_t rows, dim_t cols, cumeric_t min = -1_c, cumeric_t max = 1_c);
        static Matrix Zeros(dim_t rows, dim_t cols);
        static Matrix Ones(dim_t rows, dim_t cols);
        static Matrix Linspace(cumeric_t start, cumeric_t end, dim_t num); /* Row Vector */

        static Matrix Copy(const Matrix& mat);
        static const Matrix View(const Matrix& mat);

		/* Conversions */

		operator Vector() const;
        // operator Vector() const;

        void fill(cumeric_t value);

        dim_t rows() const { return rows_; }
        dim_t cols() const { return cols_; }
        
        Matrix row(const dim_t i) const;
        Matrix col(const dim_t i) const;
        
        dim_t size() const { return rows_ * cols_; }
        
        /* Accesors */
        cumeric_t* data() { return data_; };
        const cumeric_t* data() const { return data_; };

        cumeric_t& at(const dim_t row, const dim_t col) { return data_[get_idx(row, col)]; }
        const cumeric_t& at(const dim_t row, const dim_t col) const { return data_[get_idx(row, col)]; }

        cumeric_t& operator () (const dim_t row, const dim_t col) { return data_[get_idx(row, col)]; };
        const cumeric_t& operator () (const dim_t row, const dim_t col) const { return data_[get_idx(row, col)]; };

		Matrix slice(dim_t i, dim_t j, dim_t rows, dim_t cols) const;

        /* assingment operator */
        // Matrix& operator = (const Matrix& other);
        // Matrix& operator = (const Matrix& other);
        Matrix& set(const Matrix& other);
        Matrix& operator = (Matrix other) noexcept;

        void swap(Matrix& other);
        friend void swap(Matrix& A, Matrix& B);

        /* Inplace arithmetic operator */
        Matrix& operator += (const Matrix& other);
        Matrix& operator += (const cumeric_t& scalar); // add scalar to all elements
        Matrix& operator -= (const Matrix& other);
        Matrix& operator -= (const cumeric_t& scalar); // substract scalar form all elements
        Matrix& operator *= (const Matrix& other);
        Matrix& operator *= (const cumeric_t& scalar);
        Matrix& operator /= (const Matrix& other);
        Matrix& operator /= (const cumeric_t& scalar);
  
        /* Outplace arithmetic operator */
        friend Matrix operator + (const Matrix& A, const Matrix& B);
		friend Matrix operator + (const Matrix& mat, cumeric_t scalar);
		friend Matrix operator + (const cumeric_t scalar, const Matrix& mat);
        friend Matrix operator - (const Matrix& A, const Matrix& B);
        friend Matrix operator * (const Matrix& A, const Matrix& B);
        friend Matrix operator * (const Matrix& mat, cumeric_t scalar);
        friend Matrix operator * (const cumeric_t scalar, const Matrix& mat);
        friend Matrix operator / (const Matrix& A, const Matrix& B);
        friend Matrix operator / (const Matrix& mat, cumeric_t scalar);
        friend Matrix operator / (const cumeric_t scalar, const Matrix& mat);
		friend bool operator == (const Matrix& A, const Matrix& B);
		friend bool operator != (const Matrix& A, const Matrix& B);
        
        /* Transforming matrix dimensions */
        Matrix transpose() const;
        Matrix& transposeInPlace();
        Matrix flatten() const;
        Matrix reshape(const dim_t rows, const dim_t cols) const;
       	Matrix& reshapeInPlace(dim_t rows, dim_t cols);

        friend Matrix activation(const Matrix& mat, const char* name);
        friend Matrix activationInPlace(Matrix& mat, const char* name);

        Matrix cwiseProduct(const Matrix& other);
        Matrix& cwiseProductInPlace(const Matrix& other);

		/* Elementwise functions */
        Matrix transform(void (*func)(cumeric_t* data, const dim_t size)) const;
        Matrix& transformInPlace(void (*func)(cumeric_t* data, const dim_t size));

        Matrix transform(cumeric_t (*func)(cumeric_t x)) const;
        Matrix& transformInPlace(cumeric_t (*func)(cumeric_t x));

		Matrix sin();
		Matrix& sinInPlace();

		Matrix cos();
		Matrix& cosInPlace();

		Matrix tan();
		Matrix& tanInPlace();

		Matrix sinh();
		Matrix& sinhInPlace();

		Matrix cosh();
		Matrix& coshInPlace();

		Matrix tanh();
		Matrix& tanhInPlace();

		Matrix sqrt();
		Matrix& sqrtInPlace();

		Matrix square();
		Matrix& squareInPlace();

		Matrix exp(); // e^{x} for each element
		Matrix& expInPlace();

		Matrix log(); // ln(x) for each element
		Matrix& logInPlace();

		Matrix relu();
		Matrix& reluInPlace();

		/* idk how to name it */

        Matrix& rowwiseOpInPlace(void (*op)(cumeric_t* row, const cumeric_t* v, const dim_t cols), const cumeric_t* arr);
        Matrix& colwiseOpInPlace(void (*op)(cumeric_t* col, const cumeric_t* v, const dim_t rows), const cumeric_t* arr);

        Matrix clip(const cumeric_t min, const cumeric_t max) const;
        Matrix& clipInPlace(const cumeric_t min, const cumeric_t max);



		/* Scalar reductions */
        cumeric_t norm();
        cumeric_t squaredNorm();
        cumeric_t sum() const;
        cumeric_t sumAbs() const;
        cumeric_t mean() const;
        cumeric_t amean() const;

		/* Vector reductions */
        Matrix colwiseSum();
        Vector colwiseSum2vec();
        Matrix rowwiseSum();
        Vector rowwiseSum2vec();

        Matrix shuffleRows() const;
        Matrix& shuffleRowsInPlace();

    private:
        dim_t get_idx(const dim_t row, const dim_t col) const { return row * cols_ + col; }

        dim_t rows_ = 0;
        dim_t cols_ = 0;
        cumeric_t* data_ = nullptr;
    };

} // namespace cum
