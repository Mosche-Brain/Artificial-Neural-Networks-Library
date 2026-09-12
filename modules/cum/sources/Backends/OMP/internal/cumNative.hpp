#pragma once

#include <oneapi/mkl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <memory>
#include <mutex>
#include <oneapi/mkl/rng/device.hpp>

#include "cum/Core.hpp"

namespace cum::internal
{
    struct Context
    {

        std::once_flag initialized;

        cumeric_t* zeros;
        cumeric_t* ones;
        cumeric_t* cache;

        void setDevice(CUM_DEVICE device);

        ~Context();
        Context();
    };

    Context& getContext();

    oneapi::mkl::rng::device::philox4x32x10<1>& getRNG();

    cumeric_t* getZeros();
    cumeric_t* getOnes();
}