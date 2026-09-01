/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Świadomy Mężczyzna
 * @email       : contact@mosche.dev
 * @file        : kernel.hpp
 * @createdOn   : 09/01/26
 * @description : Implementation for opaque kernel type
 *---------------------------------------------**/

#pragma once

#include "cum/detail/vendor/oneapi/kernel.hpp"

namespace cum::experimental
{
    sycl::kernel& __kernel__::operator ()()
    {
        return kernel;
    }
}