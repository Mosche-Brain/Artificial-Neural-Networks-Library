/* Created by jaro on 7/19/26. */

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>

#include "cum/neural_primitives/opaque_types.hpp"

namespace cum::neural_primitives::handles
{
    struct __memory__ { dnnl::memory memory; };
    struct __engine__ { dnnl::engine engine; };
    struct __stream__ { dnnl::stream stream; };

    struct  __desc__  { dnnl::memory::desc desc; };
}
