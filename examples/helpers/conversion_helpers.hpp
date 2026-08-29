//
// Created by jaro on 7/17/26.
//

#include <cum/Matrix.hpp>
#include <cum/runtime.hpp>

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
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

inline void saveMatrixToFile(
    const cum::Matrix& matrix,
    const std::string& filename,
    int precision = 9)
{
    cum::runtime::sync();

    std::ofstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Unable to open matrix output file: " + filename);
    }

    file << std::setprecision(precision);
    file << "# rows " << matrix.rows() << " cols " << matrix.cols() << '\n';
    for (std::size_t row = 0; row < matrix.rows(); ++row)
    {
        for (std::size_t col = 0; col < matrix.cols(); ++col)
        {
            if (col != 0)
            {
                file << ' ';
            }
            file << static_cast<double>(matrix(row, col));
        }
        file << '\n';
    }
}