//
// Created by jaro on 6/28/26.
//

#include <oneapi/dnnl/dnnl.hpp>

#include "cum/neural_primitives/Descriptor.hpp"
#include "cum/detail/vendor/oneapi/opaque_types.hpp"

#include "cum/Tensor.hpp"

namespace cum
{
    Tensor::Tensor(Shape shape, datatype dtype, layout layout)
		: __desc__(std::make_unique<neural_primitives::Descriptor>(shape, dtype, layout)),
		  __data__(std::make_unique<neural_primitives::Memory>(*__desc__))
	{

	}

	Tensor::~Tensor()
	{

	}

	cumeric_t Tensor::sum()
    {

    }


} // cum
