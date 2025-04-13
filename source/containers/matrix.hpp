#pragma once

#include <cstdalign>
#include <string>
#include <iostream>
#include <cstdint>
#include <stdexcept>
#include <cstddef>
#include <vector>
#include <array>

template <typename T>
class Matrix
{
public:
    Matrix(size_t rows, size_t cols, T fill_value = T())
    {
        if(fill_value != 0)
        {
            this->buffer = std::vector<std::vector<T>>(rows, std::vector<T>(cols, fill_value));
            return;
        }

            this->buffer = std::vector<std::vector<T>>(rows);
    }
    
    T& at(size_t row, size_t col)
    {
        return this->buffer.at(row).at(col);
    }

    std::vector<T> row(size_t i)
    {
        return this->buffer.at(i);
    }

    std::vector<T> col(size_t i)
    {
        if(i >= this->cols_)
            throw std::out_of_range("Column index out of range");

        std::vector<T> buff;
        buff.reserve(cols_);

        for(size_t j = 0 ; j < rows_ ; j++)
        {
            buff.push_back(this->buffer.at(j).at(i));
        }
    }

    void fill(T value)
    {
        for(size_t i = 0 ; i < rows_ ; i++)
        {
            for(size_t j = 0 ; j < cols_ ; j++)
            {
                this->buffer.at(i).at(j) = value;
            }
        }
    }

    friend Matrix operator + (const Matrix &a, const Matrix &b)
    {
        if(!a.compareDimensions(b))
        {
            std::cerr << "Matrix dimensions don't match\n";
        }

        Matrix<T> temp(a.rows_, a.cols_);

        for(size_t i = 0 ; i < rows ; i++)
        {
            for(size_t j = 0 ; i < cols ; j++)
            {
                temp.at(i, j) = a.at(i, j) + b.at(i, j);
            }
        }

        return temp;
    }

    friend Matrix operator + (const Matrix &a, const T scalar)
    {
        Matrix<T> temp(a.rows_, a.cols_);

        for(size_t i = 0 ; i < rows ; i++)
        {
            for(size_t j = 0 ; i < cols ; j++)
            {
                temp.at(i, j) = a.at(i, j) + scalar;
            }
        }

        return temp;
    }

    friend Matrix operator - (const Matrix &a, const Matrix &b)
    {
        if(!a.compareDimensions(b))
        {
            std::cerr << "Matrix dimensions don't match\n";
        }

        Matrix<T> temp(a.rows_, a.cols_);

        for(size_t i = 0 ; i < rows ; i++)
        {
            for(size_t j = 0 ; i < cols ; j++)
            {
                temp.at(i, j) = a.at(i, j) - b.at(i, j);
            }
        }

        return temp;
    }

    friend Matrix operator - (const Matrix &a, const T scalar)
    {
        Matrix<T> temp(a.rows_, a.cols_);

        for(size_t i = 0 ; i < rows ; i++)
        {
            for(size_t j = 0 ; i < cols ; j++)
            {
                temp.at(i, j) = a.at(i, j) - scalar;
            }
        }

        return temp;
    }

    bool compareDimensions(const Matrix other)
    {
        return this->row == other.rows_ && this->cols_ == other.cols_;
    }
    
    size_t rows()
    {
        return this->rows_;
    }

    size_t cols()
    {
        return this->cols_;
    }

protected:
    size_t rows_;
    size_t cols_;
    std::vector<std::vector<T>> buffer;
};

/*
template <typename T>
void displayMatrix(Matrix<T>& matrix)
{
    for(size_t i = 0 ; i < matrix.rows() ; i++)
    {
        for(size_t j = 0 ; j < matrix.cols() ; j++)
        {
            //std::ostringstream element;
            //element << matrix.at(i).at(j);

            //std::cout << element << ',' << ' ';
            std::cout << matrix.at(i, j) << ',' << ' ';
        }

        std::cout << std::endl;
    }
}
*/