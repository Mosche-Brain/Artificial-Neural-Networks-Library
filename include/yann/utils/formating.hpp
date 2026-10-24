/**
 * @file formating.hpp
 * @brief String formatting utilities for printing matrix dimensions, content, and side-by-side visualizations.
 */

#pragma once

#include <iostream>
#include <string>

namespace cum { class Matrix; }

namespace yann::utils::formating
{
    /**
     * @brief Format matrix dimensions into a human-readable string (e.g. "[rows x cols]").
     * @param mat Matrix to inspect.
     * @return Formatted dimension string.
     */
    std::string show_matrix_dimensions(const cum::Matrix &mat);

    /**
     * @brief Convert matrix values into a formatted 2D table string.
     * @param mat Matrix to format.
     * @return Multi-line formatted string representing matrix contents.
     */
    std::string matrixToString(const cum::Matrix& mat);

    /**
     * @brief Format two matrices side-by-side with an arrow separator for visual comparisons.
     * @param leftMatrix Matrix on the left side of the arrow.
     * @param rightMatrix Matrix on the right side of the arrow.
     * @param tabIndent Leading spaces/tab indentation for lines (default: 0).
     * @param cellWidth Character width for each matrix cell (default: 8).
     * @return Multi-line string showing both matrices separated by an arrow.
     */
    std::string matricesWithArrowToString(
        const cum::Matrix& leftMatrix,
        const cum::Matrix& rightMatrix,
        int tabIndent = 0,
        int cellWidth = 8);
}
