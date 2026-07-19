#pragma once

#include "cum/Core.hpp"

namespace cum::memory
{
    cumeric_t* allocate(std::size_t size);
    void free(void* chunk);
    void memcopy(void* dst, const void* src, std::size_t size);
    void prefetch(void* chunk, std::size_t size);

}