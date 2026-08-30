#pragma once

#include "cum/Matrix.hpp"

namespace cum
{
    class Matrix;
    class Vector // : Matrix
    {
    public:
		Vector(dim_t lenght);
        Vector(std::size_t lenght, cumeric_t value);
        Vector() = default;
        // Vector(Matrix& M);

        operator Matrix() const;
        
        cumeric_t& operator[](const std::size_t index) { return this->data_[index]; }
        const cumeric_t operator[](const std::size_t index) const { return this->data_[index]; }
        // const cumeric_t& operator[](const std::size_t index) const { return this->data_[index]; }

        cumeric_t* data() { return data_; }

        std::size_t lenght() const { return lenght_; }
        std::size_t size()   const { return lenght_; }

        Vector& operator += (const Vector& other);
        Vector& operator += (Matrix& matrix);
        Vector& operator -= (const Vector& other);
        Vector& operator *= (const Vector& other);
        Vector& operator *= (const cumeric_t& scalar);
        Vector& operator /= (const Vector& other);

        friend Vector operator + (const Vector& v, const Vector& u);
        friend Vector operator + (const Vector& v, const Matrix& A);
        friend Vector operator - (const Vector& v, const Vector& u);
        friend Vector operator * (const Vector& v, const Vector& u);
        friend Vector operator * (const Vector& v, const cumeric_t& scalar);
        // friend Vector operator * (const Vector& v, const Matrix& A);
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
