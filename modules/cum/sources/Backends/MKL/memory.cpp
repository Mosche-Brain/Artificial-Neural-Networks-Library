#include "cum/memory.hpp"

#include "internal/context.hpp"
#include <sycl/sycl.hpp>

namespace cum::memory
{
    cumeric_t* allocate(std::size_t size)
    {
        return sycl::malloc_shared<cumeric_t>(size, internal::getQueue());
        // return sycl::malloc_device<cumeric_t>(size, internal::getQueue());
    }

    void free(void* chunk)
    {
        sycl::free(chunk, internal::getQueue());
    }

    void memcopy(void* dst, const void* src, std::size_t size)
    {
        internal::getQueue().memcpy(dst, src, size).wait();
    }

    void prefetch(void* chunk, std::size_t size)
    {
        internal::getQueue().prefetch(chunk, size).wait();
    }
}
