#include "cumMKL.hpp"
#include "cum/functions/various.hpp"
#include <algorithm>
#include <iostream>

namespace cum
{
    internal::Context::Context() // : queue(sycl::default_selector{})
    {
        // this->setDevice(CUM_DEVICE::AUTO);
        // zeros = sycl::malloc_shared<cumeric_t>(2048*2048, queue);
        // ones = sycl::malloc_shared<cumeric_t>(2048*2048, queue);

        // std::call_once(this->initialized, [](){
        //     auto device = getContext().queue.get_device();
        //     std::cout << "cum initialized, used device: " << device.get_info<sycl::info::device::name>() << '\n';
        // });
    }

    internal::Context::~Context()
    {              
        sycl::free(zeros, queue);
        sycl::free(ones, queue);
    }

    void internal::Context::setDevice(CUM_DEVICE device)
    {
        sycl::free(zeros, queue);
        sycl::free(ones, queue);
        queue.wait();

        if(device == CUM_DEVICE::AUTO)
            queue = sycl::queue(sycl::default_selector_v);
        else if(device == CUM_DEVICE::CPU)
            queue = sycl::queue(sycl::cpu_selector_v, sycl::property::queue::in_order{});
        else if(device == CUM_DEVICE::GPU)
            queue = sycl::queue(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        else
            throw std::runtime_error("Invalid device type");



        zeros = sycl::malloc_shared<cumeric_t>(2048*2048, queue);
        ones  = sycl::malloc_shared<cumeric_t>(2048*2048, queue);
        cache = sycl::malloc_shared<cumeric_t>(2048*2048, queue);

        queue.wait();

        constexpr std::size_t buffer_size = 2048 * 2048;
        constexpr std::size_t chunk_size = 65536;
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
    }

    internal::Context& internal::getContext()
    {
        static Context context{};
        return context;
    }

    sycl::queue& internal::getQueue()
    {
        return getContext().queue;
    }

    oneapi::mkl::rng::device::philox4x32x10<1>& internal::getRNG()
    {
        return getContext().rng;
    }

    cumeric_t* internal::getZeros() { return getContext().zeros; }
    cumeric_t* internal::getOnes() { return getContext().ones; }
}
