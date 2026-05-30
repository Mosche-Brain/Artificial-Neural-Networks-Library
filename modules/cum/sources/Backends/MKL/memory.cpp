#include "cum/memory.hpp"

#include "cumMKL.hpp"
#include <sycl/sycl.hpp>

namespace cum::memory
{
    cumeric_t* allocate(std::size_t size)
    {
        return sycl::malloc_shared<cumeric_t>(size, library::getQueue());
    }

    void free(void* chunk)
    {
        sycl::free(chunk, library::getQueue());
    }
}
