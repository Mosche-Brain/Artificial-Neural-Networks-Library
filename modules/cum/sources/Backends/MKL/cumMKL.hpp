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

            void setDevice(CUM_DEVICE device);

            ~Context();
            Context();
        };

        // static Context& context;

        Context& getContext();

        sycl::queue& getQueue();

        cumeric_t* getZeros();
        cumeric_t* getOnes();
    }

    // using cumeric_t = sycl::half;

} // namespace cum
