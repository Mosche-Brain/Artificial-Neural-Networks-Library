#pragma once

#include "cum/Matrix.hpp"

namespace cum
{
    class Vector // : Matrix
    {
    public:
        Vector(std::size_t lenght, cumeric_t value=0);
    
        cumeric_t& operator[](std::size_t index) { return this->data_[index]; }
        cumeric_t* data() { return data_; }

        std::size_t lenght() const { return lenght_; }

        Vector& operator +=(const Vector& other);
        Vector& operator -=(const Vector& other);
        Vector& operator *=(const Vector& other);
        Vector& operator /=(const Vector& other);

        friend Vector operator + (const Vector& v, const Vector& u);
        friend Vector operator - (const Vector& v, const Vector& u);
        friend Vector operator * (const Vector& v, const Vector& u);
        friend Vector operator / (const Vector& v, const Vector& u);
    
        Vector dot(const Vector& other);
        Vector& dotInPlace(const Vector& other);

        cumeric_t sum();

        Vector normalize();
        Vector& normalizeInPlace();

    private:
        std::size_t lenght_;
        cumeric_t* data_;
    };
} // namespace cum
