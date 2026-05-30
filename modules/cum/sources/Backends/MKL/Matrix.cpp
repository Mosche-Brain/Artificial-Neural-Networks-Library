#include "cum/Matrix.hpp"
#include "cum/Vector.hpp"
#include "cum/LinearAlgebra.hpp"
#include "cum/functions.hpp"
#include "cum/random.hpp"

#include "cumMKL.hpp"

#include <utility>

#if defined(BUILD_ENABLE_IO_OVERLOADS)
#include <iostream>
#endif

namespace cum
{
    Matrix::Matrix(size_t rows, size_t cols, cumeric_t value) : rows_(rows), cols_(cols)
    {
        data_ = sycl::malloc_shared<cumeric_t>(rows * cols, library::getQueue());
        for(size_t i = 0 ; i < rows * cols ; i++)
            data_[i] = value;
    }

    Matrix::Matrix(std::size_t rows, std::size_t cols, cumeric_t* source)
    {
        data_ = sycl::malloc_shared<cumeric_t>(rows * cols, library::getQueue());

        // sycl::memc
        for(size_t i = 0 ; i < rows * cols ; i++)
            data_[i] = source[i];
    }

    Matrix::Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_)
    {
        data_ = sycl::malloc_shared<cumeric_t>(rows_ * cols_, library::getQueue());
        library::getQueue().memcpy(data_, other.data_, rows_ * cols_ * sizeof(cumeric_t)).wait();
    }

    Matrix::Matrix(Matrix&& other) noexcept : data_(other.data_), rows_(other.rows_), cols_(other.cols_)
    {
        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }

    Matrix::~Matrix()
    {
        sycl::free(data_, library::getQueue());
    }

    Matrix Matrix::Random(std::size_t rows, std::size_t cols, cumeric_t min, cumeric_t max)
    {
        Matrix temp;
        temp.data_ = sycl::malloc_shared<cumeric_t>(rows * cols, library::getQueue());
        temp.rows_ = rows;
        temp.cols_ = cols;

        cum::random::uniform(temp.data_, rows * cols, min, max);
        
        return temp;
    }

    Matrix Matrix::Zeros(std::size_t rows, std::size_t cols)
    {
        Matrix temp;
        temp.data_ = sycl::malloc_shared<cumeric_t>(rows * cols, library::getQueue());
        temp.rows_ = rows;
        temp.cols_ = cols;

        return temp;
    }

    Matrix Matrix::Ones(std::size_t rows, std::size_t cols)
    {
        Matrix temp;
        temp.data_ = sycl::malloc_shared<cumeric_t>(rows * cols, library::getQueue());
        temp.rows_ = rows;
        temp.cols_ = cols;

        for(size_t i = 0 ; i < rows * cols ; i++)
        {
            temp.data_[i] = 1._c;
        }

        return temp;
    }

    /* ============================== Accessors ================================ */

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

    /* ============================== Asingnment operators ================================ */

    // Matrix& Matrix::operator = (const Matrix& other)
    Matrix& Matrix::set(const Matrix& other)
    {
        if (this == &other)
            return *this;

        if(data_ != nullptr)
        {
            sycl::free(data_, library::getQueue());
            data_ =        nullptr;
        }
        data_ = sycl::malloc_shared<cumeric_t>(other.rows_ * other.cols_, library::getQueue());

        library::getQueue().memcpy(other.data_, data_, other.rows_ * other.cols_ * sizeof(cumeric_t)).wait();
        return *this;
    }

    Matrix& Matrix::operator = (Matrix other) noexcept
    {
        if(data_ == nullptr)
            data_ = sycl::malloc_shared<cumeric_t>(other.rows_ * other.cols_, library::getQueue());
        swap(other);
        return *this;
    }

    void Matrix::swap(Matrix& other)
    {
        std::swap(this->data_, other.data_);
        std::swap(this->rows_, other.rows_);
        std::swap(this->cols_, other.cols_);
    }

    void swap(Matrix& A, Matrix& B)
    {
        A.swap(B);
    }

    /* ============================== Inplace arithmetic operator ================================*/

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

    Matrix& Matrix::operator /= (const Matrix& other)
    {
        // LinearAlgebra::matDivInPlace(this->data_, other.data_, this->rows_, this->cols_);
        return *this;
    }

    Matrix& Matrix::operator /= (const cumeric_t& scalar)
    {
        LinearAlgebra::scaleInPlace(this->data_, 1 / scalar, rows_ * cols_);
        return *this;
    }

    /* ============================== Outplace arithmetic operator ============================== */

    Matrix operator + (const Matrix& A, const Matrix& B)
    {
        Matrix mat(A.rows_, A.cols_);

        LinearAlgebra::matAdd(mat.data_, A.data_, B.data_, A.rows_, A.cols_);
        return mat;
    }

    Matrix operator - (const Matrix& A, const Matrix& B)
    {
        Matrix mat(A.rows_, A.cols_);

        LinearAlgebra::matSub(mat.data_, A.data_, B.data_, A.rows_, A.cols_);
        return mat;
    }

    Matrix operator * (const Matrix& A, const Matrix& B)
    {
        Matrix mat(A.rows_, B.cols_);

        LinearAlgebra::matMul(mat.data_, A.data_, B.data_, A.rows_, B.cols_, A.cols_);
        return mat; 
    }

    // Matrix operator * (const Matrix& A, const Vector& v)
    // {
    //     Matrix mat(A.rows_, B.cols_);

    //     LinearAlgebra::matMul(mat.data_, A.data_, B.data_, A.rows_, B.cols_, A.cols_);
    //     return mat; 
    // }

    Matrix operator * (const Matrix& mat, const cumeric_t& scalar)
    {
        Matrix temp(mat.rows_, mat.cols_);
        memcpy(temp.data_, mat.data_, mat.rows_ * mat.cols_ * sizeof(cumeric_t));
        
        LinearAlgebra::scaleInPlace(temp.data_, scalar, mat.rows_ * mat.cols_);
        return temp;
    }

    Matrix operator / (const Matrix& A, const Matrix& B)
    {
        Matrix mat(A.rows_, A.cols_);

        // LinearAlgebra::matDiv(mat.data_, A.data_, B.data_, A.rows_, A.cols_);
        return mat; 
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


    // Matrix Matrix::transform(void (*func)(cumeric_t* data, const std::size_t size)) const
    // {
    //     Matrix temp(rows_, cols_);
    //     memcpy(temp.data_, data_, rows_ * cols_ * sizeof(cumeric_t));
    //     temp.transformInPlace(func);
    //     return temp;
    // }


    // Matrix& Matrix::transformInPlace(void (*func)(cumeric_t* data, const std::size_t size))
    // {
    //     func(data_, rows_ * cols_);
    //     return *this;
    // }


    // Matrix Matrix::transform(cumeric_t (*func)(cumeric_t x)) const
    // {
    //     Matrix temp(rows_, cols_);
    //     memcpy(temp.data_, data_, rows_ * cols_ * sizeof(cumeric_t));
    //     temp.transformInPlace(func);
    //     return temp;
    // }

    // Matrix& Matrix::transformInPlace(cumeric_t (*func)(cumeric_t x))
    // {
    //     auto& q = library::getQueue();
    //     LinearAlgebra::transformInPlace(data_, func, rows_ * cols_);
    //     return *this;
    // }

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

    Vector Matrix::colwiseSum2vec()
    {
        Vector temp(cols_);
        LinearAlgebra::colwiseSum(temp.data(), data_, rows_, cols_);
        return temp;
    }

    Matrix Matrix::rowwiseSum()
    {
        Matrix temp(rows_, 1);
        LinearAlgebra::rowwiseSum(temp.data_, data_, rows_, cols_);
        return temp;
    }

    Vector Matrix::rowwiseSum2vec()
    {
        Vector temp(rows_);
        LinearAlgebra::rowwiseSum(temp.data(), data_, rows_, cols_);
        return temp;
    }

    Vector Matrix::flatten() const
    {
        Vector temp(rows_ * cols_);
        memcpy(temp.data(), data_, rows_ * cols_ * sizeof(cumeric_t));
        return temp;
    }

    cumeric_t Matrix::norm()
    {
        cumeric_t result;
        LinearAlgebra::norm(result, data_, rows_ * cols_);
        return result;
    }

    cumeric_t Matrix::squaredNorm()
    {
        cumeric_t result;
        LinearAlgebra::squaredNorm(result, data_, rows_ * cols_);
        return result;
    }

    Matrix activation(const Matrix& mat, const char* name)
    {
        Matrix temp(mat.rows_, mat.cols_);
        memcpy(temp.data_, mat.data_, mat.rows_ * mat.cols_ * sizeof(cumeric_t));
        // functions:: activationInPlace(temp, name);
        return temp;
    }
    Matrix activationInPlace(Matrix& mat, const char* name);
    
} // namespace cum