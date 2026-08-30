
#include "cum/Core.hpp"
#include "cum/LinearAlgebra/BLAS/level1.hpp"
#include "cum/Vector.hpp"
#include "cum/LinearAlgebra/matops.hpp"
#include "cum/LinearAlgebra/vecops.hpp"
#include "cum/LinearAlgebra.hpp"
#include "cum/functions.hpp"
#include "cum/functions/various.hpp"
#include "cum/random.hpp"

#include "internal/cumMKL.hpp"

#include <stdexcept>
#include <utility>
#include <random>

#include "cum/memory.hpp"
#include "cum/runtime.hpp"

#if defined(BUILD_ENABLE_IO_OVERLOADS)
#include <iostream>
#endif

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>

#include "cum/Matrix.hpp"

namespace cum
{
	Matrix::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols)
	{
		data_ = memory::allocate(rows * cols);
	}

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

		memory::memcopy(data_, source, rows_ * cols_ * sizeof(cumeric_t));
        // sycl::memc
        //for(size_t i = 0 ; i < rows * cols ; i++)
        //    data_[i] = source[i];
    }    

	Matrix::Matrix(std::size_t rows, std::size_t cols, const cumeric_t* source) : rows_(rows), cols_(cols)
    {
        // data_ = sycl::malloc_shared<cumeric_t>(rows * cols, internal::getQueue());
        data_ = memory::allocate(rows * cols);

        // sycl::memc
		memory::memcopy(data_, source, rows_ * cols_ * sizeof(cumeric_t)); 
		//for(size_t i = 0 ; i < rows * cols ; i++)
        //    data_[i] = source[i];
    }

    Matrix::Matrix(std::size_t rows, std::size_t cols, std::initializer_list<cumeric_t> elements) : rows_(rows), cols_(cols)
    {
        //data_ = sycl::malloc_shared<cumeric_t>(rows * cols, internal::getQueue()); 
        //internal::getQueue().copy(elements.begin(), data_, rows * cols).wait();
    	data_ = memory::allocate(rows * cols);
		memory::memcopy(data_, elements.begin(), rows * cols * sizeof(cumeric_t));
	}


    Matrix::Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_)
    {
        //data_ = sycl::malloc_shared<cumeric_t>(rows_ * cols_, internal::getQueue());
        //internal::getQueue().memcpy(data_, other.data_, rows_ * cols_ * sizeof(cumeric_t)).wait();
    	data_ = memory::allocate(rows_ * cols_);
		memory::memcopy(data_, other.data_, rows_ * cols_ * sizeof(cumeric_t));
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

        functions::various::fill(temp.data_, 0._c, rows * cols);

        return temp;
    }

    Matrix Matrix::Ones(std::size_t rows, std::size_t cols)
    {
        Matrix temp;

        temp.data_ = memory::allocate(rows * cols);
        temp.rows_ = rows;
        temp.cols_ = cols;

        functions::various::fill(temp.data_, 1._c, rows * cols);

		return temp;
    }

    Matrix Matrix::Linspace(cumeric_t start, cumeric_t end, std::size_t num)
    {
        //cumeric_t* buff = memory::allocate(num);
        //internal::getQueue().parallel_for(sycl::range<1>(num), [=](sycl::id<1> idx)
        //{
        //    const std::size_t i = idx[0];
//
//            if (num == 1)
//            {
//                buff[i] = start;
//            }
//            else
//            {
//                buff[i] = start + static_cast<cumeric_t>(i) * (end - start) / static_cast<cumeric_t>(num - 1);
//            }
//        }).wait();

        Matrix temp;

		temp.data_ = memory::allocate(num);
		temp.rows_ = 1;
		temp.cols_ = num;

		functions::various::linespace(temp.data_, start, end, num);

        return temp;
    }

	Matrix::operator Vector() const
	{
		if(rows_ == rows_ * cols_ || cols_ == rows_ * cols_)
			std::invalid_argument("Invalid dimensions");
		
		Vector temp(rows_ * cols_);
		memory::memcopy(temp.data(), this->data_, rows_ * cols_ * sizeof(cumeric_t));
		
		return temp;
	}

    void Matrix::fill(const cumeric_t value)
    {
        functions::various::fill(data_, value, rows_ * cols_);
    }

    /* ============================== Accessors ================================ */

    Matrix Matrix::row(size_t i) const
    { 
        Matrix temp(1, cols_);
		runtime::sync();
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
       
		dim_t m = this->rows_;
		dim_t n = this->cols_;
		cumeric_t* src = this->data_;
		cumeric_t* dest = temp.data_;
		internal::getQueue().parallel_for(sycl::range<1>(m), [=](sycl::id<1> idx)
		{
			dest[idx] = src[idx * n + i];
		}).wait();

        return temp;
    }

	Matrix Matrix::slice(const dim_t i, const dim_t j, const dim_t rows, const dim_t cols)
	{
		if(i + rows > rows_ || j + cols > rows_)
			std::invalid_argument("Slice exceds matrix dimensions");

		Matrix temp;

		temp.data_ = memory::allocate(rows * cols);
		temp.rows_ = rows_;
		temp.cols_ = cols_;

		dim_t src_cols = this->cols_; 
		cumeric_t* src = this->data_;
		cumeric_t* dest = temp.data_;
		internal::getQueue().parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> idx)
		{
			dim_t i_ = idx[0];
			dim_t j_ = idx[1];

			dest[i_ * cols + j_] = src[(i + i_) * src_cols + (j + j_)];
		}).wait();

		return temp;
	}

    /* ============================== Asingnment operators ================================ */

    // Matrix& Matrix::operator = (const Matrix& other)
    Matrix& Matrix::set(const Matrix& other)
    {
        if (this == &other)
        {
            return *this;
        }

        Matrix replacement(other);
        swap(replacement);
        return *this;
    }

    Matrix& Matrix::operator = (Matrix other) noexcept
    {
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
        //LinearAlgebra::matAddInPlace(this->data_, other.data_, this->rows_, this->cols_);
		LinearAlgebra::vadd(this->data_, other.data_, rows_ * cols_);
		return *this;
    }

    Matrix& Matrix::operator += (const cumeric_t& scalar)
    {
        LinearAlgebra::addInPlace(this->data_, scalar, rows_ * cols_);
		
		return *this;
    }

    Matrix& Matrix::operator -= (const Matrix& other)
    {
        LinearAlgebra::matSubInPlace(this->data_, other.data_, this->rows_, this->cols_);
        return *this;
    }

    Matrix& Matrix::operator *= (const Matrix& other)
    {
        if (cols_ != other.rows_ || other.cols_ != cols_)
        {
            throw std::invalid_argument("In-place matrix multiplication dimension mismatch");
        }

        LinearAlgebra::matMulInPlace(this->data_, other.data_, this->rows_, other.cols_, this->cols_);
        return *this;
    }

    Matrix& Matrix::operator *= (const cumeric_t& scalar)
    {
        //LinearAlgebra::scaleInPlace(this->data_, scalar, rows_ * cols_);
		blas::scal(rows_ * cols_, scalar, data_, 1);
		return *this;
    }

    Matrix& Matrix::operator /= (const Matrix& other)
    {
        // LinearAlgebra::matDivInPlace(this->data_, other.data_, this->rows_, this->cols_);
        return *this;
    }

    Matrix& Matrix::operator /= (const cumeric_t& scalar)
    {
        //LinearAlgebra::scaleInPlace(this->data_, 1 / scalar, rows_ * cols_);
		blas::scal(rows_ * cols_, 1 / scalar, data_, 1);
		return *this;
    }

    /* ============================== Outplace arithmetic operator ============================== */

    Matrix operator + (const Matrix& A, const Matrix& B)
    {
        Matrix mat(A.rows_, A.cols_);

		LinearAlgebra::vadd(mat.data_, A.data_, A.rows_ * A.cols_);
        //LinearAlgebra::matAdd(mat.data_, A.data_, B.data_, A.rows_, A.cols_);
        return mat;
    }

    Matrix operator - (const Matrix& A, const Matrix& B)
    {
        Matrix mat(A.rows_, A.cols_);

		LinearAlgebra::vsub(mat.data_, A.data_, B.data_, A.rows_ * A.cols_);
        //LinearAlgebra::matSub(mat.data_, A.data_, B.data_, A.rows_, A.cols_);
        return mat;
    }

    Matrix operator * (const Matrix& A, const Matrix& B)
    {
        if (A.cols_ != B.rows_)
        {
            throw std::invalid_argument("Matrix multiplication dimension mismatch");
        }

        Matrix mat(A.rows_, B.cols_);

        LinearAlgebra::mmul(mat.data_, A.data_, B.data_, A.rows_, B.cols_, A.cols_);
		cum::runtime::sync();
        return mat;
    }

    // Matrix operator * (const Matrix& A, const Vector& v)
    // {
    //     Matrix mat(A.rows_, B.cols_);

    //     LinearAlgebra::matMul(mat.data_, A.data_, B.data_, A.rows_, B.cols_, A.cols_);
    //     return mat; 
    // }

    Matrix operator * (const Matrix& mat, const cumeric_t scalar)
    {
        Matrix temp(mat.rows_, mat.cols_, mat.data_);
		//Matrix temp(mat.rows_, mat.cols_);
		//memory::memcopy(temp.data_, mat.data_, mat.rows_ * mat.cols_);
		//runtime::sync();
        //internal::getQueue().memcpy(
        //    temp.data_, mat.data_, mat.size() * sizeof(cumeric_t)).wait();
        LinearAlgebra::scaleInPlace(temp.data_, scalar, mat.size());
		//runtime::sync();
        return temp;
    }

    Matrix operator * (const cumeric_t scalar, const Matrix& mat)
    {
        return mat * scalar;
    }

    Matrix operator + (const Matrix& mat, const cumeric_t scalar)
    {
        Matrix temp(mat);
        temp += scalar;
        return temp;
    }

    Matrix operator + (const cumeric_t scalar, const Matrix& mat)
    {
        return mat + scalar;
    }

    Matrix operator / (const Matrix& A, const Matrix& B)
    {
        Matrix mat(A.rows_, A.cols_);
	
        LinearAlgebra::div(mat.data_, A.data_, B.data_, A.rows_ * A.cols_);
        return mat; 
    }

    Matrix operator / (const Matrix& mat, const cumeric_t& scalar)
    {
        Matrix temp(mat.rows_, mat.cols_);
        //memcpy(temp.data_, mat.data_, mat.rows_ * mat.cols_ * sizeof(cumeric_t));
        //LinearAlgebra::scaleInPlace(temp.data_, 1 /scalar, mat.rows_ * mat.cols_);
		memory::memcopy(temp.data_, mat.data_, mat.rows_ * mat.cols_ * sizeof(cumeric_t));
		blas::scal(temp.rows_ * temp.cols_, scalar, temp.data_, 1);

		return temp;
    }

    bool operator == (const Matrix& A, const Matrix& B)
    {
        // Yes, no runtime dimensions checks 💪

        auto policy = oneapi::dpl::execution::make_device_policy(internal::getQueue());\
        return std::equal(policy, A.data(), A.data() + A.rows() * B.cols(), B.data());
    }

    Matrix Matrix::transpose() const
    {
        Matrix temp(cols_, rows_);
        //LinearAlgebra::transpose(temp.data_, data_, rows_, cols_);
		LinearAlgebra::mtrans(temp.data_, data_, rows_, cols_);
		cum::runtime::sync();
		return temp;
    }

    Matrix& Matrix::transposeInPlace()
    {
        //LinearAlgebra::transposeInPlace(data_, rows_, cols_);
		LinearAlgebra::mtrans(data_, rows_, cols_);
		cum::runtime::sync();
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
        Matrix temp(*this);
        temp.transformInPlace(func);
        return temp;
    }

    Matrix& Matrix::transformInPlace(void (*func)(cumeric_t* data, const std::size_t size))
    {
        if (func == nullptr)
        {
            throw std::invalid_argument("Matrix transform requires a function");
        }
        internal::getQueue().wait();
        func(data_, size());
        internal::getQueue().wait();
        return *this;
    }

    Matrix Matrix::transform(cumeric_t (*func)(cumeric_t x)) const
    {
        Matrix temp(*this);
        temp.transformInPlace(func);
        return temp;
    }

    Matrix& Matrix::transformInPlace(cumeric_t (*func)(cumeric_t x))
    {
        if (func == nullptr)
        {
            throw std::invalid_argument("Matrix transform requires a function");
        }
        internal::getQueue().wait();
        for (std::size_t i = 0; i < size(); ++i)
        {
            data_[i] = func(data_[i]);
        }
        internal::getQueue().wait();
        return *this;
    }

    Matrix Matrix::sin()
    {
        Matrix temp(*this);
        return temp.sinInPlace();
    }

    Matrix& Matrix::sinInPlace()
    {
        functions::trigonometric::sin(data_, data_, size());
        internal::getQueue().wait();
        return *this;
    }

    Matrix Matrix::cos()
    {
        Matrix temp(*this);
        return temp.cosInPlace();
    }

    Matrix& Matrix::cosInPlace()
    {
        functions::trigonometric::cos(data_, data_, size());
        internal::getQueue().wait();
        return *this;
    }

    Matrix Matrix::tan()
    {
        Matrix temp(*this);
        return temp.tanInPlace();
    }

    Matrix& Matrix::tanInPlace()
    {
        functions::trigonometric::tan(data_, data_, size());
        internal::getQueue().wait();
        return *this;
    }

    Matrix Matrix::sinh()
    {
        Matrix temp(*this);
        return temp.sinhInPlace();
    }

    Matrix& Matrix::sinhInPlace()
    {
        functions::hyperbolic::sinh(data_, data_, size());
        internal::getQueue().wait();
        return *this;
    }

    Matrix Matrix::cosh()
    {
        Matrix temp(*this);
        return temp.coshInPlace();
    }

    Matrix& Matrix::coshInPlace()
    {
        functions::hyperbolic::cosh(data_, data_, size());
        internal::getQueue().wait();
        return *this;
    }

    Matrix Matrix::tanh()
    {
        Matrix temp(*this);
        return temp.tanhInPlace();
    }

    Matrix& Matrix::tanhInPlace()
    {
        functions::hyperbolic::tanh(data_, data_, size());
        internal::getQueue().wait();
        return *this;
    }

    Matrix Matrix::sqrt()
    {
        Matrix temp(*this);
        return temp.sqrtInPlace();
    }

    Matrix& Matrix::sqrtInPlace()
    {
        auto& q = internal::getQueue();
        q.parallel_for(sycl::range<1>(size()), [data = data_](sycl::id<1> index)
        {
            data[index] = sycl::sqrt(data[index]);
        }).wait();
        return *this;
    }

    Matrix Matrix::square()
    {
        Matrix temp(*this);
        return temp.squareInPlace();
    }

    Matrix& Matrix::squareInPlace()
    {
        auto& q = internal::getQueue();
        q.parallel_for(sycl::range<1>(size()), [data = data_](sycl::id<1> index)
        {
            data[index] *= data[index];
        }).wait();
        return *this;
    }

    Matrix Matrix::exp()
    {
        Matrix temp(*this);
        return temp.expInPlace();
    }

    Matrix& Matrix::expInPlace()
    {
        functions::exponential::exp(data_, data_, size());
        internal::getQueue().wait();
        return *this;
    }

    Matrix Matrix::log()
    {
        Matrix temp(*this);
        return temp.logInPlace();
    }

    Matrix& Matrix::logInPlace()
    {
        functions::exponential::log(data_, data_, size());
        internal::getQueue().wait();
        return *this;
    }

    Matrix Matrix::relu()
    {
        Matrix temp(*this);
        return temp.reluInPlace();
    }

    Matrix& Matrix::reluInPlace()
    {
        functions::linear_units::relu(data_, data_, size());
        internal::getQueue().wait();
        return *this;
    }

    Matrix Matrix::cwiseProduct(const Matrix& other)
    {
        Matrix temp(rows_, cols_);
        //LinearAlgebra::cwiseProduct(temp.data_, data_, other.data_, rows_ * cols_);
		LinearAlgebra::vmul(temp.data_, this->data_, other.data_, rows_ * cols_);
		return temp;
    }

    Matrix& Matrix::cwiseProductInPlace(const Matrix& other)
    {
        //LinearAlgebra::cwiseProductInPlace(data_, data_, rows_ * cols_);
		LinearAlgebra::vmul(data_, other.data_, rows_ * cols_);
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

    Matrix Matrix::flatten() const
    {
        //Vector temp(rows_ * cols_);
        //memcpy(temp.data(), data_, rows_ * cols_ * sizeof(cumeric_t));
		Matrix temp;

		temp.data_ = memory::allocate(rows_ * cols_);
		temp.rows_ = 1;
		temp.cols_ = rows_ * cols_;

		memory::memcopy(temp.data_, this->data_, rows_ * cols_ * sizeof(cumeric_t));

        return temp;
    }

    Matrix Matrix::reshape(const std::size_t rows, const std::size_t cols) const
    {
        return {rows, cols, data_};
    }

	Matrix& Matrix::reshapeInPlace(const dim_t rows, const dim_t cols)
	{
		if(rows * cols != rows_ * cols)
			std::invalid_argument("rows * cols != rows_ * cols_");

		this->rows_ = rows;
		this->cols_ = cols;

		return *this;
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

    cumeric_t Matrix::sumAbs() const
    {
        cumeric_t result = 0;
        cum::LinearAlgebra::asum(result, data_, rows_ * cols_);
        return result;
    }
    cumeric_t Matrix::sum() const
    {
        cumeric_t result = 0;
        cum::LinearAlgebra::sum(result, data_, rows_ * cols_);
        return result;
    }

    cumeric_t Matrix::mean() const
    {
        return this->sum() / static_cast<cumeric_t>(rows_ * cols_);
    }
    cumeric_t Matrix::amean() const
    {
        return this->sumAbs() / static_cast<cumeric_t>(rows_ * cols_);
    }

    cumeric_t Matrix::squaredNorm()
    {
        cumeric_t result;
        LinearAlgebra::squaredNorm(result, data_, rows_ * cols_);
        return result;
    }


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
