#pragma once
#include <iostream>

namespace cum { class Matrix; }

namespace YANN::Utils::logs
{
    std::string show_matrix_dimensions(const cum::Matrix &mat);
}