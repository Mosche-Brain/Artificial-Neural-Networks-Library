#include "cum/Matrix.hpp"
#include "cum/LinearAlgebra.hpp"

#include "cumMKL.hpp"

#include <utility>

namespace cum
{
    Matrix::Matrix(size_t rows, size_t cols, cumeric_t value) : rows_(rows), cols_(cols)
    {
        data = sycl::malloc_shared<cumeric_t>(rows * cols, library::getQueue());
        for(size_t i = 0 ; i < rows * cols ; i++)
            data[i] = value;
    }

    Matrix::~Matrix()
    {
        sycl::free(data, library::getQueue());
    }

    cumeric_t& Matrix::at(const size_t row, const size_t col)
    {
        return data[index(row, col)];
    }

    std::size_t Matrix::index(const size_t row, const size_t col)
    {
        return row * cols_ + col;
    }

    Matrix& Matrix::operator += (const Matrix& other)
    {
        LinearAlgebra::matAddInPlace(this->data, other.data, this->rows_, this->cols_);
    }

    Matrix& Matrix::operator -= (const Matrix& other)
    {
        LinearAlgebra::matSubInPlace(this->data, other.data, this->rows_, this->cols_);
    }

    Matrix& Matrix::operator *= (const Matrix& other)
    {
        LinearAlgebra::matMulInPlace(this->data, other.data, this->rows_, other.cols_, this->cols_);
    }

    Matrix operator + (const Matrix& A, const Matrix& B)
    {
        Matrix mat(A.rows_, A.cols_);
        LinearAlgebra::matAdd(mat.data, A.data, B.data, A.rows_, A.cols_);
        return mat;
    }

    Matrix operator * (const Matrix& A, const Matrix& B)
    {
        Matrix mat(A.rows_, B.cols_);

        LinearAlgebra::matMul(mat.data, A.data, B.data, A.rows_, B.cols_, A.cols_);

        return mat; 
    }

    Matrix Matrix::transpose()
    {
        Matrix temp(cols_, rows_);
        LinearAlgebra::transpose(temp.data, data, rows_, cols_);
        return temp;
    }

    void Matrix::transposeInPlace()
    {
        LinearAlgebra::transposeInPlace(data, rows_, cols_);
        std::swap<std::size_t>(cols_, rows_);
    }

    Matrix Matrix::cwiseProduct(const Matrix& other)
    {
        
    }

    void Matrix::cwiseProductInPlace()
    {
        // LinearAlgebra:
    }
} // namespace cum

