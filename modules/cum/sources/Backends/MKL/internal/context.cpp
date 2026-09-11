#include "context.hpp"
#include "cum/functions/various.hpp"
#include <algorithm>
#include <iostream>
#include <print>
#include <sycl/context.hpp>
#include <sycl/device.hpp>

namespace cum
{
    internal::Context::Context(DEVICE device) // : queue(sycl::default_selector{})
    {
        switch (device)
        {
            case DEVICE::AUTO:
            {
                this->queue = sycl::queue(sycl::default_selector_v);
                break;
            }
            case DEVICE::CPU:
            {
                this->queue = sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order{});
                break;
            }
            case DEVICE::GPU:
            {
                this->queue = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
                break;
            }
            default:
            {
                this->queue = sycl::queue(sycl::default_selector_v);
            }
        }

        constexpr dim_t buffer_size = 2048 * 2048;
        constexpr dim_t chunk_size = 65536;

        cache_size = buffer_size;
        zeros = sycl::malloc_shared<cumeric_t>(buffer_size, queue);
        ones  = sycl::malloc_shared<cumeric_t>(buffer_size, queue);
        cache = sycl::malloc_shared<cumeric_t>(buffer_size, queue);

        queue.wait();


        queue.fill(zeros, static_cast<cumeric_t>(0), buffer_size);
        queue.fill(ones, static_cast<cumeric_t>(1), buffer_size);

        // sycl::ext::oneapi::experimental::fill(queue, zeros, static_cast<cumeric_t>(0), buffer_size);
        // sycl::ext::oneapi::experimental::fill(queue, ones, static_cast<cumeric_t>(1), buffer_size);

        this->device = queue.get_device();
        this->context = queue.get_context();

        this->engine = dnnl::sycl_interop::make_engine(queue.get_device(), queue.get_context());
        this->stream = dnnl::sycl_interop::make_stream(engine, queue);

        std::cout << "device: " << queue.get_device().get_info<sycl::info::device::name>() << '\n';;
        std::cout << "vendor: " << queue.get_device().get_info<sycl::info::device::vendor>() << '\n';;
        // std::cout << "platform: " << queue.get_device().get_info<sycl::info::device::device_type>() << '\n';;
    }

    internal::Context::~Context()
    {              
        sycl::free(zeros, queue);
        sycl::free(ones, queue);
        sycl::free(cache, queue);
    }

    void internal::Context::setDevice(DEVICE device)
    {
        sycl::free(zeros, queue);
        sycl::free(ones, queue);
        queue.wait();

        if(device == DEVICE::AUTO)
            queue = sycl::queue(sycl::default_selector_v);
        else if(device == DEVICE::CPU)
            queue = sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order{});
        else if(device == DEVICE::GPU)
            queue = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        else
            throw std::runtime_error("Invalid device type");

        constexpr std::size_t buffer_size = 2048 * 2048;
        constexpr std::size_t chunk_size = 65536;

        zeros = sycl::malloc_shared<cumeric_t>(buffer_size, queue);
        ones  = sycl::malloc_shared<cumeric_t>(buffer_size, queue);
        cache = sycl::malloc_shared<cumeric_t>(buffer_size, queue);

        queue.wait();

        auto& q = queue;
        const auto initialize = [&q, buffer_size, chunk_size](cumeric_t* buffer, cumeric_t value)
        {
            for (std::size_t offset = 0; offset < buffer_size; offset += chunk_size)
            {
                const std::size_t count = std::min(chunk_size, buffer_size - offset);
                q.parallel_for(
                    sycl::range<1>(count),
                    [buffer, value, offset](sycl::id<1> index)
                    {
                        buffer[offset + index] = value;
                    }).wait();
            }
        };
        initialize(zeros, static_cast<cumeric_t>(0));
        initialize(ones, static_cast<cumeric_t>(1));
        // queue = sycl::queue(selected_device);
        std::cout << "Selected device: " << queue.get_device().get_info<sycl::info::device::name>() << '\n';;
        std::cout << "Used backend: " << queue.get_device().get_info<sycl::info::device::name>() << '\n';;
        std::println("cache size: {}", buffer_size);
    }

    internal::Context& internal::context(DEVICE device)
    {
        static Context context(device);
        return context;
    }

    sycl::queue& internal::getQueue()
    {
        return context().queue;
    }

    oneapi::mkl::rng::device::philox4x32x10<1>& internal::getRNG()
    {
        return context().rng;
    }

    cumeric_t* internal::getZeros() { return context().zeros; }
    cumeric_t* internal::getOnes() { return context().ones; }

    sycl::context& internal::sycl_context()
    {
        return context().context;
    }

	sycl::queue& internal::queue()
	{
		return context().queue;
	}

	sycl::device& internal::device()
	{
		return context().device;
	}

    dnnl::engine& internal::engine()
    {
        return context().engine;
    }

    dnnl::stream& internal::stream()
    {
        return context().stream;
    }

    const cumeric_t* internal::zeros()
    {
        return context().zeros;
    }

    const cumeric_t* internal::ones()
    {
        return context().zeros;
    }

    cumeric_t* internal::cache()
    {
        return context().cache;
    }
}
