#pragma once

#include <oneapi/mkl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <memory>
#include <mutex>

#include "cum/Core.hpp"

namespace cum::internal
{
    struct Context
    {
        sycl::queue queue;
        sycl::context context;
        sycl::device device;
        dnnl::engine engine;
        dnnl::stream stream;

        std::once_flag initialized;

        cumeric_t* zeros;
        cumeric_t* ones;
        cumeric_t* cache;

        void setDevice(CUM_DEVICE device);

        ~Context();
        Context();
    };

    Context& getContext();

    sycl::context& getSyclContext();
    sycl::device& device();
    sycl::queue& getQueue();
    dnnl::engine& getEngine();
    dnnl::stream& getStream();

    cumeric_t* getZeros();
    cumeric_t* getOnes();
}