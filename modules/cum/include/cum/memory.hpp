#pragma once

#include "cum/Core.hpp"

namespace cum::memory
{
    cumeric_t* allocate(std::size_t size);
    void free(void* chunk);
}