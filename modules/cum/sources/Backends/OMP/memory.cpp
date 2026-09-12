/*
 * @author: jaro
 * @name:   memory
 * @file:   modules/cum/sources/Backends/OMP/memory.cpp
 * @date:   12 September 2026 20:46:55
 */

#include <cstdlib>

#include "cum/Core.hpp"

namespace cum::memory
{
    cumeric_t* allocate(dim_t size)
    {
        // return sycl::malloc_device<cumeric_t>(size, internal::getQueue());
    }

    void free(void* chunk)
    {
        std::free(chunk);
    }

    void memcopy(void* dst, const void* src, dim_t size)
    {
        std::memcpy(dst, src, size);
    }

    void prefetch(void* chunk, dim_t size)
    {
        // Nothing, there is no GPU support on OMP backend
    }
}
