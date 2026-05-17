#include "logs.hpp"

#include <cum/Matrix.hpp>

namespace YANN::Utils::logs
{
    std::string show_matrix_dimensions(const cum::Matrix& mat)
    {
        return std::to_string(mat.rows()) + "x" + std::to_string(mat.cols());
    }
}