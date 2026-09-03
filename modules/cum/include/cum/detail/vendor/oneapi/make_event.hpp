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
    class make_event
    {
    public:
        static experimental::__event__ create(sycl::event&& event);
        // {
        //     return experimental::__event__(std::make_unique<experimental::__event__::Impl>(event));
        // }
    };
}