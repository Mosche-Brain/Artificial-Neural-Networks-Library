/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Świadomy Mężczyzna
 * @email       : 
 * @file        : 
 * @createdOn   : 09/03/26 
 * @description : 
 *---------------------------------------------**/

#pragma once

#include <memory>
#include <utility>

#include <sycl/event.hpp>

#include "cum/experimental/__event__.hpp"

namespace cum::detail
{
    class event_handler
    {
    public:
        static experimental::__event__ create(sycl::event&& event);
        static sycl::event& handle(experimental::__event__& event);
    };
}