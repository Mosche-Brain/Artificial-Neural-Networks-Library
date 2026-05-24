#include "logs.hpp"

#include <cum/Matrix.hpp>

namespace YANN::Utils::logs
{
    std::string show_matrix_dimensions(const cum::Matrix& mat)
    {
        return std::to_string(mat.rows()) + "x" + std::to_string(mat.cols());
    }

    std::string matrixToString(const cum::Matrix& mat)
    {
        std::string string;
        for(size_t i = 0 ; i < mat.rows() ; i++)
        {
            for(size_t j = 0 ; j < mat.cols() ; j++)
                string += std::to_string(mat(i, j)) + '\t';
            string += '\n';
        }
        return string;
    }
}