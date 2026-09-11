#pragma once

#include <oneapi/dnnl/dnnl_common.hpp>
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
    	explicit Context(DEVICE device=DEVICE::AUTO);
    	~Context();


        std::once_flag initialized;

        dim_t cache_size;
        cumeric_t* zeros;
        cumeric_t* ones;
        cumeric_t* cache;

        void setDevice(DEVICE device);

    	sycl::queue queue;
    	sycl::context context;
    	sycl::device device;
    	dnnl::engine engine;
    	dnnl::stream stream;
    	oneapi::mkl::rng::device::philox4x32x10<1> rng;
    };

    Context& context(DEVICE device=DEVICE::AUTO);

    sycl::context& getSyclContext();
    sycl::device& device();
    sycl::queue& getQueue();
    dnnl::engine& getEngine();
    dnnl::stream& getStream();

    oneapi::mkl::rng::device::philox4x32x10<1>& getRNG();

    cumeric_t* getZeros();
    cumeric_t* getOnes();

    const cumeric_t* zeros();
    const cumeric_t* ones();
	cumeric_t* cache();

    sycl::context& sycl_context();
	sycl::queue& queue();
	dnnl::engine& engine();
	dnnl::stream& stream();
}
