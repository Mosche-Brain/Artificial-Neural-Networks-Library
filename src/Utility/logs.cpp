#include "logs.hpp"

#include <cum/Matrix.hpp>

#include <iomanip>

namespace yann::Utils::logs
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

    std::string matricesWithArrowToString(
        const cum::Matrix& leftMatrix,
        const cum::Matrix& rightMatrix,
        int tabIndent,
        int cellWidth)
    {
        const int rows = leftMatrix.rows();
        const int cols = leftMatrix.cols();

        // if (rightMatrix.rows() != rows || rightMatrix.cols() != cols) {
        //     throw std::invalid_argument("Matrices must have the same dimensions.");
        // }

        std::ostringstream out;
        const std::string indent(tabIndent, '\t');

        for (int row = 0; row < rows; ++row) {
            out << indent;

            out << "[ ";
            for (int col = 0; col < cols; ++col) {
                out << std::setw(cellWidth)
                    << leftMatrix(row, col);
            }
            out << " ]";

            if (row == (rows - 1) / 2) {
                out << "  --->  ";
            } else {
                out << "        ";
            }

            out << "[ ";
            for (int col = 0; col < cols; ++col) {
                out << std::setw(cellWidth)
                    << rightMatrix(row, col);
            }
            out << " ]";

            out << '\n';
        }

        return out.str();

    }
}