#pragma once
#include <iostream>

namespace cum { class Matrix; }

namespace yann::utils::formating
{
    std::string show_matrix_dimensions(const cum::Matrix &mat);

    std::string matrixToString(const cum::Matrix& mat);

    std::string matricesWithArrowToString(
        const cum::Matrix& leftMatrix,
        const cum::Matrix& rightMatrix,
        int tabIndent = 0,
        int cellWidth = 8);
}