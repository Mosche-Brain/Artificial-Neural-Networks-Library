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

#include "cum/detail/vendor/oneapi/event_handler.hpp"

#include "cum/experimental/__event__.hpp"

namespace cum::experimental 
{
    struct __event__::Impl
    {
        sycl::event event;
        explicit Impl(sycl::event&& e) : event(std::move(e)) {}
    };

    __event__::__event__() noexcept : handle(nullptr) {}

    __event__::__event__(std::unique_ptr<Impl> handle) noexcept : handle(std::move(handle)) {}

    __event__::__event__(__event__&&) noexcept = default;
    
    __event__& __event__::operator = (__event__&&) noexcept = default;

    __event__::~__event__() = default;

    void __event__::wait()
    {
        if(handle != nullptr) handle->event.wait();
    }
}

namespace cum::detail
{
    experimental::__event__ event_handler::create(sycl::event&& event)
    {
        return experimental::__event__(std::make_unique<experimental::__event__::Impl>(std::move(event)));
    }

    sycl::event& event_handler::handle(experimental::__event__& event)
    {
        return event.handle->event;
    }
}

// Note: I must get better understanding of r-value refferences