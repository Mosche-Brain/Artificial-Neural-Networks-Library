#include "cumMKL.hpp"

#include <iostream>

namespace cum
{
    library::Context::Context() : queue(sycl::cpu_selector_v) 
    {
        zeros = sycl::malloc_shared<cumeric_t>(2048*2048, queue);
        ones = sycl::malloc_shared<cumeric_t>(2048*2048, queue);
    }

    library::Context& library::getContext()
    {
        static Context context;
        return context;
    }

    sycl::queue& library::getQueue()
    {
        std::call_once(getContext().initialized, [](){
            auto device = getContext().queue.get_device();
            std::cout << "device: " << device.get_info<sycl::info::device::name>() << '\n';
        });
        return getContext().queue;
    }

    cumeric_t* library::getZeros() { return getContext().zeros; }
    cumeric_t* library::getOnes() { return getContext().ones; }
}