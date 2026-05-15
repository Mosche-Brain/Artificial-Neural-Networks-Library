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

            cumeric_t* zeros;
            cumeric_t* ones;

            Context();
        };

        Context& getContext();

        sycl::queue& getQueue();

        cumeric_t* getZeros();
        cumeric_t* getOnes();
    }

    // using cumeric_t = sycl::half;

} // namespace cum
