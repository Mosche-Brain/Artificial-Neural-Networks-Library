/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Swiadomy-Mężczyzna
 * @email       : contact@mosche.dev
 * @createdOn   : 09/01/26
 * @description : Declaration of opaque __event__ structure
 *----------------------------------------------**/

// #include "cum/detail/vendor/oneapi/event.hpp"

#include <memory>
#include <sycl/event.hpp>

#include "cum/experimental/__event__.hpp"

namespace cum::experimental 
{
    struct __event__::Impl
    {
        sycl::event event;
        explicit Impl(sycl::event&& e) : event(std::move(e)) {}
    };

    // __event__::__event__(__event__&& other) noexcept : handle(other.handle)
    // {
    //     other.handle = nullptr;
    // }

    __event__::__event__(std::unique_ptr<Impl> handle) noexcept : __handle__(std::move(handle)) {}

    __event__::__event__(__event__&&) noexcept = default;
    
    __event__& __event__::operator = (__event__&&) noexcept = default;

    
    __event__::~__event__() = default;

    void __event__::wait()
    {
        handle->event.wait();
    }
}

// Note: I must get better understanding of r-value refferences