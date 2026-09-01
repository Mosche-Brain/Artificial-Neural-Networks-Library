/**-----------------------------------------------
 * ?                    ABOUT
 * @author      : Swiadomy-Mężczyzna
 * @email       : contact@mosche.dev
 * @createdOn   : 09/01/26
 * @description : Definition of opaque __event__ structure
 *----------------------------------------------**/

#include <sycl/event.hpp>

namespace cum::experimental 
{
    struct __event__
    {
        sycl::event event;

        sycl::event& operator ()();
        
        void wait();
    };
} 

