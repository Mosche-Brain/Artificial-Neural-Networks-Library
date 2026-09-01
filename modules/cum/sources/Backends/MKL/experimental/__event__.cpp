/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Swiadomy-Mężczyzna
 * @email       : contact@mosche.dev
 * @createdOn   : 9/1/26
 * @description : Declaration of opaque __event__ structure
 *----------------------------------------------**/

#include "cum/detail/vendor/oneapi/event.hpp"

namespace cum::experimental
{
    sycl::event& __event__::operator ()()
    {
        return event;
    }

    void __event__::wait()
    {
        event.wait();
    }

    void wait_for_event(__event__& event)
    {
        event.wait();
    }
}

/*
#include <sycl/event.hpp>

namespace cum::experimental 
{
    struct __event__
    {
        sycl::event event;
        
        sycl::event& operator ()() { return event; }
        void wait() { event.wait(); }
        
    };
} 
*/