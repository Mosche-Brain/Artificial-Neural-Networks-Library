#pragma once

#include "cum/Matrix.hpp"

namespace cum
{
    class Vector // : Matrix
    {
    public:
        Vector(std::size_t lenght, cumeric_t value=0);
    
        cumeric_t& operator[](std::size_t index) { this->data[index]; }

        Vector operator +=(const Vector& other);
        Vector operator -=(const Vector& other);
        Vector operator *=(const Vector& other);
        Vector operator /=(const Vector& other);

        friend Vector operator + (const Vector& v, const Vector& u);
        friend Vector operator - (const Vector& v, const Vector& u);
        friend Vector operator * (const Vector& v, const Vector& u);
        friend Vector operator / (const Vector& v, const Vector& u);
    
        Vector dot(const Vector& other);
    private:
        cumeric_t* data;
    };
} // namespace cum
