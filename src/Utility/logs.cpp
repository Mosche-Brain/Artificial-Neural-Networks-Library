#include "logs.hpp"


namespace ANN::Utils::logs
{
    std::string show_matrix_dimensions(const Eigen::MatrixXf& mat)
    {
        return std::to_string(mat.rows()) + "x" + std::to_string(mat.cols());
    }
}