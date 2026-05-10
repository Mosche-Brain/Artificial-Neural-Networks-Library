#include "cumMKL.hpp"

#include <iostream>

namespace cum
{
    library::Context::Context() : queue(sycl::cpu_selector_v) {}

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
}