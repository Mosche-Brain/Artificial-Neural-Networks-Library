#include "cum/Matrix.hpp"
#include "cum/Vector.hpp"
#include "cum/LinearAlgebra.hpp"
#include "cum/functions.hpp"
#include "cum/random.hpp"

#include "internal/cumMKL.hpp"

#include <utility>
#include <random>

#include "cum/memory.hpp"

#if defined(BUILD_ENABLE_IO_OVERLOADS)
#include <iostream>
#endif

namespace cum
{
    Matrix::Matrix(size_t rows, size_t cols, cumeric_t value) : rows_(rows), cols_(cols)
    {
        data_ = memory::allocate(rows * cols);
        // for(size_t i = 0 ; i < rows * cols ; i++)
        //     data_[i] = value;
        cum::functions::various::fill(data_, value, rows * cols);
    }

    Matrix::Matrix(std::size_t rows, std::size_t cols, cumeric_t* source) : rows_(rows), cols_(cols)
    {
        // data_ = sycl::malloc_shared<cumeric_t>(rows * cols, internal::getQueue());
        data_ = memory::allocate(rows * cols);

        // sycl::memc
        for(size_t i = 0 ; i < rows * cols ; i++)
            data_[i] = source[i];
    }

    Matrix::Matrix(std::size_t rows, std::size_t cols, std::initializer_list<cumeric_t> elements) : rows_(rows), cols_(cols)
    {
        data_ = sycl::malloc_shared<cumeric_t>(rows * cols, internal::getQueue());

        // for(size_t i = 0 ; i < rows * cols ; i++)
        //     data_[i] = elements.;
        internal::getQueue().copy(elements.begin(), data_, rows * cols).wait();
    }


    Matrix::Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_)
    {
        data_ = sycl::malloc_shared<cumeric_t>(rows_ * cols_, internal::getQueue());
        internal::getQueue().memcpy(data_, other.data_, rows_ * cols_ * sizeof(cumeric_t)).wait();
    }

    Matrix::Matrix(Matrix&& other) noexcept : data_(other.data_), rows_(other.rows_), cols_(other.cols_)
    {
        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }

    Matrix::~Matrix()
    {
        // sycl::free(data_, internal::getQueue());
        memory::free(data_);
    }

    Matrix Matrix::Random(std::size_t rows, std::size_t cols, cumeric_t min, cumeric_t max)
    {
        Matrix temp;
        temp.data_ = memory::allocate(rows * cols);
        temp.rows_ = rows;
        temp.cols_ = cols;

        cum::random::uniform(temp.data_, rows * cols, min, max);
        
        return temp;
    }

    Matrix Matrix::Zeros(std::size_t rows, std::size_t cols)
    {
        Matrix temp;
        temp.data_ = memory::allocate(rows * cols);
        temp.rows_ = rows;
        temp.cols_ = cols;

        return temp;
    }

    Matrix Matrix::Ones(std::size_t rows, std::size_t cols)
    {
        Matrix temp;
        // temp.data_ = sycl::malloc_shared<cumeric_t>(rows * cols, internal::getQueue());
        temp.data_ = memory::allocate(rows * cols);
        temp.rows_ = rows;
        temp.cols_ = cols;

        for(size_t i = 0 ; i < rows * cols ; i++)
        {
            temp.data_[i] = 1._c;
        }

        return temp;
    }

    Matrix Matrix::Linspace(cumeric_t start, cumeric_t end, std::size_t num)
    {


        // cumeric_t* buff = sycl::malloc_shared<cumeric_t>(num, internal::getQueue());
        cumeric_t* buff = memory::allocate(num);
        internal::getQueue().parallel_for(sycl::range<1>(num), [=](sycl::id<1> idx)
        {
            const std::size_t i = idx[0];

            if (num == 1)
            {
                buff[i] = start;
            }
            else
            {
                buff[i] = start + static_cast<cumeric_t>(i) * (end - start) / static_cast<cumeric_t>(num - 1);
            }
        }).wait();

        Matrix temp(1, num, buff);
        // temp.data_ = sycl::malloc_shared<cumeric_t>(num, internal::getQueue());
        // temp.rows_ = 1;
        // temp.cols_ = num;
        // copy buff to temp matrix data


        // internal::getQueue().submit([=](sycl::handler& h){
        //
        //     h.parallel_for(sycl::range<1>(num), [=](sycl::id<1> i)
        //     {
        //         if (num == 1) {
        //             temp.data_[i] = start;
        //         } else {
        //             temp.data_[i] = start + static_cast<cumeric_t>(i) *
        //                         (end - start) / static_cast<cumeric_t>(num - 1);
        //         }
        // }).wait();

        return temp;
    }

    void Matrix::fill(cumeric_t value)
    {
        functions::various::fill(data_, value, rows_ * cols_);
    }

    /* ============================== Accessors ================================ */

    Matrix Matrix::row(size_t i) const
    { 
        Matrix temp(1, cols_);
        internal::getQueue().memcpy(
            temp.data_,
            data_ + i * cols_,
            cols_ * sizeof(cumeric_t)
        ).wait();
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
            // sycl::free(data_, internal::getQueue());
            memory::free(data_);
            data_ = nullptr;
        }
        data_ = memory::allocate(other.rows_ * other.cols_);

        // internal::getQueue().memcpy(other.data_, data_, other.rows_ * other.cols_ * sizeof(cumeric_t)).wait();
        memory::memcopy(other.data_, data_, rows_ * cols_ * sizeof(cumeric_t));
        return *this;
    }

    Matrix& Matrix::operator = (Matrix other) noexcept
    {
        if(data_ == nullptr)
            data_ = cum::memory::allocate(other.rows_ * other.cols_);
            // data_ = sycl::malloc_shared<cumeric_t>(other.rows_ * other.cols_, internal::getQueue());
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
    //     auto& q = internal::getQueue();
    //     LinearAlgebra::transformInPlace(data_, func, rows_ * cols_);
    //     return *this;
    // }
    
    Matrix Matrix::transform(void (*func)(cumeric_t* data, const std::size_t size)) const
    {
        Matrix temp(rows_, cols_);
        memcpy(temp.data_, data_, rows_ * cols_ * sizeof(cumeric_t));
        temp.transformInPlace(func);
        return temp;
    }

    Matrix& Matrix::transformInPlace(void (*func)(cumeric_t* data, const std::size_t size))
    {
        // cum::functions::transformInPlace(data_, func, rows_ * cols_);
    }

    Matrix Matrix::transform(cumeric_t (*func)(cumeric_t x)) const
    {

    }

    Matrix& Matrix::transformInPlace(cumeric_t (*func)(cumeric_t x))
    {}

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
            // cumeric_t* row_begin = data_ + (i * cols_ * sizeof(cumeric_t));
            cumeric_t* row_begin = data_ + (i * cols_);
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


    Matrix Matrix::clip(const cumeric_t min, const cumeric_t max) const
    {
        Matrix temp(rows_, cols_);
        cum::functions::various::clip(temp.data(), data_, min, max, rows_ * cols_);
        return temp;
    }

    Matrix& Matrix::clipInPlace(const cumeric_t min, const cumeric_t max)
    {
        cum::functions::various::clipInPlace(data_, min, max, rows_ * cols_);
        return *this;
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

    Matrix Matrix::shuffleRows() const
    {
        Matrix temp(*this);
        temp.shuffleRowsInPlace();
        return temp;
    }

    Matrix& Matrix::shuffleRowsInPlace()
    {
        if (rows_ <= 1 || cols_ == 0)
            return *this;

        std::vector<std::size_t> indices(rows_);

        for (std::size_t i = 0; i < rows_; ++i)
            indices[i] = i;

        std::random_device rd;
        std::mt19937 gen(rd());

        std::shuffle(indices.begin(), indices.end(), gen);

        Matrix temp(rows_, cols_);

        for (std::size_t newRow = 0; newRow < rows_; ++newRow)
        {
            std::size_t oldRow = indices[newRow];

            std::memcpy(
                temp.data_ + newRow * cols_,
                data_ + oldRow * cols_,
                cols_ * sizeof(cumeric_t)
            );
        }

        std::memcpy(
            data_,
            temp.data_,
            rows_ * cols_ * sizeof(cumeric_t)
        );

        return *this;
    }
    
} // namespace cum