/**----------------------------------------------
 * ?                    ABOUT
 * @author      : Świadomy Mężczyzna
 * @email       : contact@mosche.dev
 * @file        : kernel.hpp
 * @createdOn   : 09/01/26
 * @description : Opaque structure definition for kernel type
 *---------------------------------------------**/

#pragma once

#include <sycl/kernel.hpp>

namespace cum::experimental
{
    struct __kernel__
    {
        sycl::kernel kernel;

        sycl::kernel& operator ()();
    };
}