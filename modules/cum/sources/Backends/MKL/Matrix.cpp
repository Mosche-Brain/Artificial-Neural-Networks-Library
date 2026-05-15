#include "cum/Matrix.hpp"
#include "cum/LinearAlgebra.hpp"

#include "cumMKL.hpp"

#include <utility>

namespace cum
{
    Matrix::Matrix(size_t rows, size_t cols, cumeric_t value) : rows_(rows), cols_(cols)
    {
        data_ = sycl::malloc_shared<cumeric_t>(rows * cols, library::getQueue());
        for(size_t i = 0 ; i < rows * cols ; i++)
            data_[i] = value;
    }

    Matrix::~Matrix()
    {
        sycl::free(data_, library::getQueue());
    }

    Matrix Matrix::row(size_t i) const
    { 
        Matrix temp(1, cols_); memcpy(temp.data_, data_ + i * rows_ * sizeof(cumeric_t), cols_ * sizeof(cumeric_t)); 
        return temp;
    }

    Matrix Matrix::col(size_t i) const
    {
        Matrix temp(rows_, 1); 
        for(size_t j = 0 ; j < rows_ ; j++)
        {
            temp.data_[j] = data_[j * cols_ + i];
        }
        return temp;
    }


    Matrix& Matrix::operator += (const Matrix& other)
    {
        LinearAlgebra::matAddInPlace(this->data_, other.data_, this->rows_, this->cols_);
        return *this;
    }

    Matrix& Matrix::operator -= (const Matrix& other)
    {
        LinearAlgebra::matSubInPlace(this->data_, other.data_, this->rows_, this->cols_);
        return *this;
    }

    Matrix& Matrix::operator *= (const Matrix& other)
    {
        LinearAlgebra::matMulInPlace(this->data_, other.data_, this->rows_, other.cols_, this->cols_);
        return *this;
    }

    Matrix& Matrix::operator *= (const cumeric_t& scalar)
    {
        LinearAlgebra::scaleInPlace(this->data_, scalar, rows_ * cols_);
        return *this;
    }

    Matrix& Matrix::operator /= (const cumeric_t& scalar)
    {
        LinearAlgebra::scaleInPlace(this->data_, 1 / scalar, rows_ * cols_);
        return *this;
    }

    Matrix operator + (const Matrix& A, const Matrix& B)
    {
        Matrix mat(A.rows_, A.cols_);

        LinearAlgebra::matAdd(mat.data_, A.data_, B.data_, A.rows_, A.cols_);
        return mat;
    }

    Matrix operator * (const Matrix& A, const Matrix& B)
    {
        Matrix mat(A.rows_, B.cols_);

        LinearAlgebra::matMul(mat.data_, A.data_, B.data_, A.rows_, B.cols_, A.cols_);
        return mat; 
    }

    Matrix operator * (const Matrix& mat, const cumeric_t& scalar)
    {
        Matrix temp(mat.rows_, mat.cols_);
        memcpy(temp.data_, mat.data_, mat.rows_ * mat.cols_ * sizeof(cumeric_t));
        
        LinearAlgebra::scaleInPlace(temp.data_, scalar, mat.rows_ * mat.cols_);
        return temp;
    }

    Matrix operator / (const Matrix& mat, const cumeric_t& scalar)
    {
        Matrix temp(mat.rows_, mat.cols_);
        memcpy(temp.data_, mat.data_, mat.rows_ * mat.cols_ * sizeof(cumeric_t));
        
        LinearAlgebra::scaleInPlace(temp.data_, 1 /scalar, mat.rows_ * mat.cols_);
        return temp;
    }

    Matrix Matrix::transpose()
    {
        Matrix temp(cols_, rows_);
        LinearAlgebra::transpose(temp.data_, data_, rows_, cols_);
        return temp;
    }

    Matrix& Matrix::transposeInPlace()
    {
        LinearAlgebra::transposeInPlace(data_, rows_, cols_);
        std::swap<std::size_t>(cols_, rows_);
        return *this;
    }


    Matrix Matrix::transform(void (*func)(cumeric_t* data, const std::size_t size)) const
    {
        Matrix temp(rows_, cols_);
        memcpy(temp.data_, data_, rows_ * cols_ * sizeof(cumeric_t));
        temp.transformInPlace(func);
        return temp;
    }


    Matrix& Matrix::transformInPlace(void (*func)(cumeric_t* data, const std::size_t size))
    {
        func(data_, rows_ * cols_);
        return *this;
    }

    Matrix Matrix::cwiseProduct(const Matrix& other)
    {
        Matrix temp(rows_, cols_);
        LinearAlgebra::cwiseProduct(temp.data_, data_, other.data_, rows_ * cols_);
        return temp;
    }

    Matrix& Matrix::cwiseProductInPlace()
    {
        LinearAlgebra::cwiseProductInPlace(data_, data_, rows_ * cols_);
        return *this;
    }

    Matrix& Matrix::rowwiseOpInPlace(void (*op)(cumeric_t* row, const cumeric_t* v, const std::size_t cols), const cumeric_t* arr)
    {
        for(size_t i = 0 ; i < rows_ ; i++)
        {
            cumeric_t* row_begin = data_ + (i * cols_ * sizeof(cumeric_t));
            op(row_begin, arr, cols_);
        }
        return *this;
    }

    Matrix& Matrix::colwiseOpInPlace(void (*op)(cumeric_t* col, const cumeric_t* v, const std::size_t rows), const cumeric_t* arr)
    {
        // for(size_t i = 0 ; i < rows_ ; i+=cols_)


        return *this;
    }

    Matrix Matrix::colwiseSum()
    {
        Matrix temp(1, cols_);
        LinearAlgebra::colwiseSum(temp.data_, data_, rows_, cols_);
        return temp;
    }

    Matrix Matrix::rowwiseSum()
    {
        Matrix temp(rows_, 1);
        LinearAlgebra::rowwiseSum(temp.data_, data_, rows_, cols_);
        return temp;
    }

    
} // namespace cum