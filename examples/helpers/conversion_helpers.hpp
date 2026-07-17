//
// Created by jaro on 7/17/26.
//

#include <cum/Matrix.hpp>

#include <vector>

template <typename T>
std::vector<T> toStdVector(const cum::Matrix& matrix)
{
    // we cannot directly copy memory because cum type may be FP16 or something else than FP64, copy with loop (nested)
    std::vector<T> result(matrix.size());
    for (std::size_t i = 0 ; i < matrix.rows() ; ++i)
    {
        for (std::size_t j = 0 ; j < matrix.cols() ; ++j)
        {
            result[i * matrix.cols() + j] = static_cast<T>(matrix(i, j));
        }
    }
    return result;
}