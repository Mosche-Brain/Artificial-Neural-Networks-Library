//
// Created by jaro on 6/28/26.
//

#include "cum/Tensor.hpp"
#include "cum/neural_primitives/tensor_descriptor.hpp"

#include <oneapi/dnnl/dnnl.hpp>

namespace cum
{
    Tensor::Tensor(Shape shape, neural_primitives::tensor_descriptor::dtype dtype, neural_primitives::tensor_descriptor::layout)
		: __desc__(std::make_unique<neural_primitives::tensor_descriptor>(shape, dtype, layout)),
		  __data__(std::make_unique<neural_primitives::Memory>(*__desc__)) 
	{

	}

	Tensor::~Tensor()
	{

	}


} // cum
