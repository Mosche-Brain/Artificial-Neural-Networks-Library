//
// Created by jaro on 9/10/26.
//

#ifndef YANN_CONVERSION_HELPERS_HPP
#define YANN_CONVERSION_HELPERS_HPP

#include <oneapi/dnnl/dnnl.hpp>

#include "cum/Core.hpp"

namespace cum
{
    dnnl::memory::data_type dnnl_data_type(datatype dtype);

    datatype cum_data_type(dnnl::memory::data_type dtype);

    dnnl::memory::format_tag dnnl_format_tag(layout format);

    layout cum_format_tag(dnnl::memory::format_tag format);

}

#endif //YANN_CONVERSION_HELPERS_HPP
