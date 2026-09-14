#pragma once

#include "cum/Core.hpp"

namespace cum::memory
{
    cumeric_t* allocate(dim_t size);

    void* allocate(dim_t size, datatype dtype);

    void free(void* chunk);
    void memcopy(void* dst, const void* src, dim_t size);
    void prefetch(void* chunk, dim_t size);

}