#include "cumMKL.hpp"

#include <iostream>

namespace cum
{
    library::Context::Context() // : queue(sycl::default_selector{})
    {
        // this->setDevice(CUM_DEVICE::AUTO);
        // zeros = sycl::malloc_shared<cumeric_t>(2048*2048, queue);
        // ones = sycl::malloc_shared<cumeric_t>(2048*2048, queue);

        // std::call_once(this->initialized, [](){
        //     auto device = getContext().queue.get_device();
        //     std::cout << "cum initialized, used device: " << device.get_info<sycl::info::device::name>() << '\n';
        // });
    }

    library::Context::~Context()
    {
        sycl::free(zeros, queue);
        sycl::free(ones, queue);
    }

    void library::Context::setDevice(CUM_DEVICE device)
    {
        sycl::free(zeros, queue);
        sycl::free(ones, queue);
        queue.wait();

        if(device == CUM_DEVICE::AUTO)
            queue = sycl::queue(sycl::default_selector_v);
        else if(device == CUM_DEVICE::CPU)
            queue = sycl::queue(sycl::cpu_selector_v);
        else if(device == CUM_DEVICE::GPU)
            queue = sycl::queue(sycl::gpu_selector_v);
        else
            throw std::runtime_error("Invalid device type");

        zeros = sycl::malloc_shared<cumeric_t>(2048*2048, queue);
        ones  = sycl::malloc_shared<cumeric_t>(2048*2048, queue);
        queue.wait();
        // queue = sycl::queue(selected_device);
        std::cout << "Selected device: " << queue.get_device().get_info<sycl::info::device::name>() << '\n';
    }

    library::Context& library::getContext()
    {
        static Context context{};
        return context;
    }

    sycl::queue& library::getQueue()
    {
        return getContext().queue;
    }

    cumeric_t* library::getZeros() { return getContext().zeros; }
    cumeric_t* library::getOnes() { return getContext().ones; }
}