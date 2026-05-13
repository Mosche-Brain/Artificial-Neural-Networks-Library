#include "cum/Vector.hpp"

#include "cum/LinearAlgebra.hpp"
#include "cumMKL.hpp"

namespace cum
{
    Vector::Vector(std::size_t lenght, cumeric_t value) : lenght_(lenght)
    {
        data_ = sycl::malloc_shared<cumeric_t>(lenght, library::getQueue());
    }


    Vector Vector::operator +=(const Vector& other)
    {
        LinearAlgebra::addInPlace(this->data_, other.data_, lenght_);
    }

    Vector Vector::operator -=(const Vector& other)
    {
        LinearAlgebra::subInPlace(this->data_, other.data_, lenght_);
    }

    Vector Vector::operator *=(const Vector& other)
    {

    }

    Vector Vector::operator /=(const Vector& other)
    {

    }


    Vector operator + (const Vector& v, const Vector& u)
    {
        
    }

    Vector operator - (const Vector& v, const Vector& u)
    {

    }

    Vector operator * (const Vector& v, const Vector& u)
    {

    }

    Vector operator / (const Vector& v, const Vector& u)
    {

    }


    Vector dot(const Vector& other)
    {

    }   

} // namespace cum
