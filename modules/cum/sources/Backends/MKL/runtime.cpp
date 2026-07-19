//
// Created by jaro on 7/19/26.
//

#include "cum/runtime.hpp"

#include <sycl/sycl.hpp>

#include "internal/cumMKL.hpp"

namespace cum::runtime
{
    /* opaque structures definitions */
    struct Device { sycl::device device_; };
    struct Queue  { sycl::queue  queue_;  };


    void sync()
    {
        internal::getQueue().wait();
    }

    // void set_default_sync_mode
}