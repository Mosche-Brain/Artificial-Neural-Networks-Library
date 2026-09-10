//
// Created by jaro on 6/28/26.
//

#include <oneapi/dnnl/dnnl.hpp>


#include "cum/runtime.hpp"
#include "cum/memory.hpp"
#include "cum/neural_primitives/Descriptor.hpp"
#include "cum/detail/vendor/oneapi/opaque_types.hpp"
#include "cum/detail/vendor/oneapi/conversion_helpers.hpp"

#include "internal/context.hpp"

#include "cum/Tensor.hpp"

#include "cum/functions.hpp"

/* This implementation have a lot of redundant code */

namespace cum
{
    Tensor::Tensor(Shape shape, datatype dtype, layout layout)
		: __desc__(std::make_unique<neural_primitives::Descriptor>(shape, dtype, layout)),
		  __data__(std::make_unique<neural_primitives::Memory>(*__desc__))
	{

	}

    Tensor::Tensor(const Tensor& tensor)
    {

    }

    Tensor::Tensor(Tensor&& tensor) noexcept : __desc__(std::move(tensor.__desc__)), __data__(std::move(tensor.__data__))
    {

    }

    Tensor::~Tensor()
	{

	}

	dim_t Tensor::size() const
    {
		return __desc__->size();
    }

	dim_t Tensor::dims() const
    {
		return __desc__->ndims();
    }

	dim_t Tensor::lenght() const
    {
		return __desc__->size() / datatype_size(__desc__->type());
    }

	Shape Tensor::shape() const
    {
	    return __desc__->shape();
    }

	layout Tensor::format() const
    {
	    return __desc__->format();
    }

	datatype Tensor::type() const
    {
	    return __desc__->type();
    }

	cumeric_t Tensor::sum()
    {
    	cumeric_t* sum_buff = memory::allocate(1);
    	sum_buff[0] = 0;

    	dnnl::memory::desc sum_desc {
    		Shape(this->dims(), 1), dnnl_data_type(default_type), dnnl::memory::format_tag::any
    	};

    	dnnl::memory sum_memory = dnnl::sycl_interop::make_memory(
    		sum_desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, sum_buff
    	);

    	dnnl::reduction::primitive_desc pd(
			internal::engine(),
			dnnl::algorithm::reduction_sum,
			__desc__->handle().desc,
			sum_desc,
			0.0f,   // p
			0.0f    // eps
		);

    	dnnl::reduction(pd).execute(
			internal::stream(),
			{
				{DNNL_ARG_SRC, __data__->handle().memory},
				{DNNL_ARG_DST, sum_memory}
			}
		);

    	internal::stream().wait();

    	cumeric_t result = sum_buff[0];
    	memory::free(sum_buff);

    	return result;
    }

	Tensor& Tensor::fill(cumeric_t scalar)
	{
    	cumeric_t* ptr = static_cast<cumeric_t*>(
			__data__->handle().memory.get_data_handle()
		);

    	const dim_t count = __desc__->handle().desc.get_size() / sizeof(cumeric_t);

		functions::various::fill(ptr, scalar, count);
    	runtime::sync();

    	return *this;
	}

	Tensor Tensor::multiply(const Tensor& tensor)
    {
    	Tensor C(shape(), type(), format());

    	dnnl::binary::primitive_desc primitive_desc {
    		internal::engine(),
			dnnl::algorithm::binary_mul,
			__desc__->handle().desc,
			tensor.__desc__->handle().desc,
			C.__desc__->handle().desc,
		};

    	dnnl::binary(primitive_desc).execute(
			internal::stream(),
			{
				{ DNNL_ARG_SRC_0, __data__->handle().memory },
				{ DNNL_ARG_SRC_1, tensor.__data__->handle().memory },
				{ DNNL_ARG_DST, C.__data__->handle().memory }
			}
		);

    	internal::stream().wait();
    	return C;
    }

	Tensor operator + (const Tensor& A, const Tensor& B)
    {
    	Tensor C(A.shape(), A.type(), A.format());

	    dnnl::binary::primitive_desc primitive_desc {
			internal::engine(),
	    	dnnl::algorithm::binary_add,
	    	A.__desc__->handle().desc,
	    	B.__desc__->handle().desc,
	    	C.__desc__->handle().desc,
	    };

    	dnnl::binary(primitive_desc).execute(
    		internal::stream(),
    		{
				{ DNNL_ARG_SRC_0, A.__data__->handle().memory },
				{ DNNL_ARG_SRC_1, B.__data__->handle().memory },
				{ DNNL_ARG_DST, C.__data__->handle().memory }
    		}
    	);

    	internal::stream().wait();
    	return C;
    }

	Tensor operator - (const Tensor& A, const Tensor& B)
    {
    	Tensor C(A.shape(), A.type(), A.format());

	    dnnl::binary::primitive_desc primitive_desc {
			internal::engine(),
	    	dnnl::algorithm::binary_sub,
	    	A.__desc__->handle().desc,
	    	B.__desc__->handle().desc,
	    	C.__desc__->handle().desc,
	    };

    	dnnl::binary(primitive_desc).execute(
    		internal::stream(),
    		{
				{ DNNL_ARG_SRC_0, A.__data__->handle().memory },
				{ DNNL_ARG_SRC_1, B.__data__->handle().memory },
				{ DNNL_ARG_DST, C.__data__->handle().memory }
    		}
    	);

    	internal::stream().wait();
    	return C;
    }

	Tensor operator * (const Tensor& A, const Tensor& B)
    {
    	Tensor C(A.shape(), A.type(), A.format());

	    dnnl::matmul::primitive_desc primitive_desc {
			internal::engine(),
	    	A.__desc__->handle().desc,
	    	B.__desc__->handle().desc,
	    	C.__desc__->handle().desc,
	    };

    	dnnl::matmul(primitive_desc).execute(
    		internal::stream(),
    		{
				{ DNNL_ARG_SRC, A.__data__->handle().memory },
				{ DNNL_ARG_WEIGHTS, B.__data__->handle().memory },
				{ DNNL_ARG_DST, C.__data__->handle().memory }
    		}
    	);

    	internal::stream().wait();
    	return C;
    }

	Tensor operator / (const Tensor& A, const Tensor& B)
    {
    	Tensor C(A.shape(), A.type(), A.format());

	    dnnl::binary::primitive_desc primitive_desc {
			internal::engine(),
	    	dnnl::algorithm::binary_div,
	    	A.__desc__->handle().desc,
	    	B.__desc__->handle().desc,
	    	C.__desc__->handle().desc,
	    };

    	dnnl::binary(primitive_desc).execute(
    		internal::stream(),
    		{
				{ DNNL_ARG_SRC_0, A.__data__->handle().memory },
				{ DNNL_ARG_SRC_1, B.__data__->handle().memory },
				{ DNNL_ARG_DST, C.__data__->handle().memory }
    		}
    	);

    	internal::stream().wait();
    	return C;
    }

	Tensor operator + (const Tensor& tensor, cumeric_t scalar)
    {
    	Tensor C(tensor.shape(), tensor.type(), tensor.format());

    	cumeric_t* scalar_buff = memory::allocate(1);
    	scalar_buff[0] = scalar;

    	dnnl::memory::desc scalar_desc {
    		Shape(tensor.dims(), 1), dnnl_data_type(default_type), dnnl::memory::format_tag::any
    	};

    	dnnl::memory scalar_memory = dnnl::sycl_interop::make_memory(
    		scalar_desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, scalar_buff
    	);

    	dnnl::binary::primitive_desc primitive_desc {
    		internal::engine(),
			dnnl::algorithm::binary_add,
			tensor.__desc__->handle().desc,
			scalar_desc,
			C.__desc__->handle().desc,
		};

    	dnnl::binary(primitive_desc).execute(
			internal::stream(),
			{
				{ DNNL_ARG_SRC_0, tensor.__data__->handle().memory },
				{ DNNL_ARG_SRC_1, scalar_memory },
				{ DNNL_ARG_DST, C.__data__->handle().memory }
			}
		);

    	internal::stream().wait();
    	memory::free(scalar_buff);

    	return C;
    }

	Tensor operator + (const cumeric_t scalar, const Tensor& tensor)
    {
	    return tensor + scalar;
    }

	Tensor operator - (const Tensor& tensor, cumeric_t scalar)
    {
    	Tensor C(tensor.shape(), tensor.type(), tensor.format());

    	cumeric_t* scalar_buff = memory::allocate(1);
    	scalar_buff[0] = scalar;

    	dnnl::memory::desc scalar_desc {
    		Shape(tensor.dims(), 1), dnnl_data_type(default_type), dnnl::memory::format_tag::any
    	};

    	dnnl::memory scalar_memory = dnnl::sycl_interop::make_memory(
    		scalar_desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, scalar_buff
    	);

    	dnnl::binary::primitive_desc primitive_desc {
    		internal::engine(),
			dnnl::algorithm::binary_sub,
			tensor.__desc__->handle().desc,
			scalar_desc,
			C.__desc__->handle().desc,
		};

    	dnnl::binary(primitive_desc).execute(
			internal::stream(),
			{
				{ DNNL_ARG_SRC_0, tensor.__data__->handle().memory },
				{ DNNL_ARG_SRC_1, scalar_memory },
				{ DNNL_ARG_DST, C.__data__->handle().memory }
			}
		);

    	internal::stream().wait();
    	memory::free(scalar_buff);

    	return C;
    }

	Tensor operator - (const cumeric_t scalar, const Tensor& tensor)
    {
    	return tensor - scalar;
    }


	Tensor operator * (const Tensor& tensor, cumeric_t scalar)
    {
    	Tensor C(tensor.shape(), tensor.type(), tensor.format());

    	cumeric_t* scalar_buff = memory::allocate(1);
    	scalar_buff[0] = scalar;

    	dnnl::memory::desc scalar_desc {
    		Shape(tensor.dims(), 1), dnnl_data_type(default_type), dnnl::memory::format_tag::any
    	};

    	dnnl::memory scalar_memory = dnnl::sycl_interop::make_memory(
    		scalar_desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, scalar_buff
    	);

    	dnnl::binary::primitive_desc primitive_desc {
    		internal::engine(),
			dnnl::algorithm::binary_mul,
			tensor.__desc__->handle().desc,
			scalar_desc,
			C.__desc__->handle().desc,
		};

    	dnnl::binary(primitive_desc).execute(
			internal::stream(),
			{
				{ DNNL_ARG_SRC_0, tensor.__data__->handle().memory },
				{ DNNL_ARG_SRC_1, scalar_memory },
				{ DNNL_ARG_DST, C.__data__->handle().memory }
			}
		);

    	internal::stream().wait();
    	memory::free(scalar_buff);

    	return C;
    }

	Tensor operator * (const cumeric_t scalar, const Tensor& tensor)
    {
    	return tensor * scalar;
    }

	Tensor operator / (const Tensor& tensor, cumeric_t scalar)
    {
    	Tensor C(tensor.shape(), tensor.type(), tensor.format());

    	cumeric_t* scalar_buff = memory::allocate(1);
    	scalar_buff[0] = scalar;

    	dnnl::memory::desc scalar_desc {
    		Shape(tensor.dims(), 1), dnnl_data_type(default_type), dnnl::memory::format_tag::any
    	};

    	dnnl::memory scalar_memory = dnnl::sycl_interop::make_memory(
    		scalar_desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, scalar_buff
    	);

    	dnnl::binary::primitive_desc primitive_desc {
    		internal::engine(),
			dnnl::algorithm::binary_div,
			tensor.__desc__->handle().desc,
			scalar_desc,
			C.__desc__->handle().desc,
		};

    	dnnl::binary(primitive_desc).execute(
			internal::stream(),
			{
				{ DNNL_ARG_SRC_0, tensor.__data__->handle().memory },
				{ DNNL_ARG_SRC_1, scalar_memory },
				{ DNNL_ARG_DST, C.__data__->handle().memory }
			}
		);

    	internal::stream().wait();
    	memory::free(scalar_buff);

    	return C;
    }

	Tensor operator / (const cumeric_t scalar, const Tensor& tensor)
    {
    	return tensor / scalar;
    }


} // cum
