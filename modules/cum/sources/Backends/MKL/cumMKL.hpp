#pragma once

#include <oneapi/mkl.hpp>
#include <memory>
#include <mutex>

#include "cum/Core.hpp"

namespace cum
{
    namespace library
    {
        struct Context
        {
            sycl::queue queue;
            std::once_flag initialized;

            Context();
        };

        Context& getContext();

        sycl::queue& getQueue();
    }

    // using cumeric_t = sycl::half;

} // namespace cum
