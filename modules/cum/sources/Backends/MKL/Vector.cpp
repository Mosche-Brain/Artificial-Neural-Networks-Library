#include "cum/Vector.hpp"
#include "cum/Matrix.hpp"

#include "cum/LinearAlgebra.hpp"
#include "cumMKL.hpp"

namespace cum
{
    Vector::Vector(std::size_t lenght, cumeric_t value) : lenght_(lenght)
    {
        data_ = sycl::malloc_shared<cumeric_t>(lenght, library::getQueue());
    }

    Vector::operator Matrix() const
    {
        Matrix temp(lenght_, 1);
        for(size_t i = 0 ; i < lenght_ ; i++)
            temp.at(i, 0) = data_[i];
        return temp;
    }

    Vector& Vector::operator += (const Vector& other)
    {
        LinearAlgebra::addInPlace(this->data_, other.data_, lenght_);
        return *this;
    }

    Vector& Vector::operator += (Matrix& matrix)
    {
        LinearAlgebra::addInPlace(this->data_, matrix.data(), lenght_);
        return *this;
    }

    Vector& Vector::operator -=(const Vector& other)
    {
        LinearAlgebra::subInPlace(this->data_, other.data_, lenght_);
        return *this;
    }

    Vector& Vector::operator *= (const Vector& other)
    {
        LinearAlgebra::cwiseProductInPlace(this->data_, other.data_, lenght_);
        return *this;                                                                                                                      
    }

    Vector& Vector::operator /= (const Vector& other)
    {
        LinearAlgebra::divInPlace (this->data_, other.data_, lenght_);
        return *this;
    }

    Vector operator + (const Vector& v, const Vector& u)
    {
        Vector result(v.lenght_);
        LinearAlgebra::add(result.data_, v.data_, u.data_, v.lenght_);
        return result;
    }

    Vector operator - (const Vector& v, const Vector& u)
    {
        Vector result(v.lenght_);
        LinearAlgebra::sub(result.data_, v.data_, u.data_, v.lenght_);
        return result;
    }

    Vector operator * (const Vector& v, const Vector& u)
    {
        Vector result(v.lenght_);
        LinearAlgebra::cwiseProduct(result.data_, v.data_, u.data_, v.lenght_);
        return result;
    }

    Vector operator / (const Vector& v, const Vector& u)
    {
        Vector result(v.lenght_);
        LinearAlgebra::div(result.data_, v.data_, u.data_, v.lenght_);
        return result;
    }

    Vector Vector::dot(const Vector& other)
    {
        Vector result(1);
        // LinearAlgebra::dot(result.data_, data_, other.data_, lenght_);
        return result;
    }

    Vector& Vector::dotInPlace(const Vector& other)
    {
        // LinearAlgebra::dot(data_, data_, other.data_, lenght_);
        return *this;
    }

    cumeric_t Vector::sum()
    {
        cumeric_t result;
        LinearAlgebra::sum(result, data_, lenght_);
        return result;
    }

    Vector Vector::normalize()
    {
        Vector result(lenght_);
        LinearAlgebra::normalize(result.data_, data_, lenght_);
        return result;
    }

    Vector& Vector::normalizeInPlace()
    {
        LinearAlgebra::normalizeInPlace(data_, lenght_);
        return *this;
    }

} // namespace cum
