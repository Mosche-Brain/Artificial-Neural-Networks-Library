/* Created by jaro on 7/19/26. */

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>

#include "cum/neural_primitives/opaque_types.hpp"

namespace cum::neural_primitives
{
    struct Memory { dnnl::memory memory; };
    struct Engine { dnnl::engine engine; };
    struct Stream { dnnl::stream stream; };

}
