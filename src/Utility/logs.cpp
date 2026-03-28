#include "logs.hpp"


namespace YANN::Utils::logs
{
    std::string show_matrix_dimensions(const matrix_t& mat)
    {
        return std::to_string(mat.rows()) + "x" + std::to_string(mat.cols());
    }
}