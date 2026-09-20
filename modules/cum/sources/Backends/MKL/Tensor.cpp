//
// Created by jaro on 6/28/26.
//

#include <cstdint>
#include <print>

#include <oneapi/dnnl/dnnl.hpp>

#include "cum/Core.hpp"
#include "cum/memory.hpp"
#include "cum/runtime.hpp"
#include "cum/datatypes.hpp"
#include "cum/functions.hpp"
#include "cum/neural_primitives/Memory.hpp"
#include "cum/neural_primitives/Descriptor.hpp"
#include "cum/neural_primitives/elementwise.hpp"
#include "cum/detail/vendor/oneapi/opaque_types.hpp"
#include "cum/neural_primitives/elementwise_diffs.hpp"
#include "cum/neural_primitives/tensor_operations.hpp"
#include "cum/detail/vendor/oneapi/conversion_helpers.hpp"

#include "internal/context.hpp"


#include "cum/Tensor.hpp"

/*
 * This implementation has a lot of redundant code.
 * We plan to move significant part to other .cpp files
 */

namespace cum
{

	int axis_position(std::size_t rank, Axis axis)
	{
		switch(rank)
		{
			case 1:
			{
					return axis == Axis::Width ? 0 : -1;
			}
			case 2:
			{
				switch(axis)
				{
					case Axis::Batches:
					case Axis::Rows:  return 0;
					case Axis::Channels:
					case Axis::Cols: return 1;
					default:             return -1;
				}
			}
			case 3:
			{
				switch(axis)
				{
					case Axis::Batches:  return 0;
					case Axis::Channels: return 1;
					case Axis::Width:    return 2;
					default:             return -1;
				}
			}
			case 4:
			{
				switch(axis)
				{
					case Axis::Batches:  return 0;
					case Axis::Channels: return 1;
					case Axis::Height:   return 2;
					case Axis::Width:    return 3;
					default:             return -1;
				}
			}
			case 5:
			{
				switch(axis)
				{
					case Axis::Batches:  return 0;
					case Axis::Channels: return 1;
					case Axis::Depth:    return 2;
					case Axis::Height:   return 3;
					case Axis::Width:    return 4;
					default:             return -1;
				}
			}
			default:
				return -1;
		}
	}

	/**------------------------------------------------------------------------------------------------
	 *                                         Constructors
	 *------------------------------------------------------------------------------------------------**/

	Tensor::Tensor(const Shape& shape, datatype dtype, layout layout)
		: _desc_(std::make_unique<neural_primitives::Descriptor>(shape, dtype, layout)),
		  _memr_(std::make_unique<neural_primitives::Memory>(*_desc_))
	{

	}

	Tensor::Tensor(const Tensor& tensor)
	{
		std::println("copy contructor called");
		_desc_ = std::make_unique<neural_primitives::Descriptor>(tensor.shape(), tensor.type(), tensor.format());

		std::println("rank: {}", tensor.rank()); // 2
		std::println("format: {}", static_cast<unsigned char>(tensor.format())); // 1 (ANY)
		// dispatch_datatype(this->type(), [&]<typename T>(){
		// 	data() = sycl::malloc_shared<T>(tensor.lenght(), internal::device(), internal::sycl_context());
		// });
		// _memr_ = std::make_unique<neural_primitives::Memory>(*_desc_, data());
		_memr_ = std::make_unique<neural_primitives::Memory>(*_desc_); // <--- This line

		internal::queue().memcpy(data(), tensor.data(), tensor.lenght() * datatype_size(this->type()));

	}

	Tensor::Tensor(Tensor&& tensor) noexcept : _desc_(std::move(tensor._desc_)), _memr_(std::move(tensor._memr_))
	{

	}


	Tensor::Tensor(const cumeric_t value, datatype dtype, layout layout) : Tensor(Shape{1}, dtype, layout)
	{
		fill(value);
	}

	Tensor::Tensor(dim_t lenght, datatype dtype, layout layout) : Tensor(Shape{lenght}, dtype, layout)
	{
	}

	Tensor::Tensor(dim_t rows, dim_t cols, datatype dtype , layout layout) : Tensor(Shape{rows, cols}, dtype, layout)
	{
	}

	Tensor::Tensor(dim_t axis0, dim_t axis1, dim_t axis2, datatype dtype, layout layout) : Tensor(Shape{axis0, axis1, axis2}, dtype, layout)
	{
	}

	Tensor::Tensor(dim_t axis0, dim_t axis1, dim_t axis2, dim_t axis3, datatype dtype, layout layout) : Tensor(Shape{axis0, axis1, axis2, axis3}, dtype, layout)
	{

	}

	Tensor::Tensor(dim_t axis0, dim_t axis1, dim_t axis2, dim_t axis3, dim_t axis4, datatype dtype, layout layout) : Tensor(Shape{axis0, axis1, axis2, axis3, axis4}, dtype, layout)
	{

	}

	Tensor::~Tensor()
	{

	}

	/**------------------------------------------------------------------------------------------------
	 *                                         Fabriques
	 *------------------------------------------------------------------------------------------------**/

	Tensor Tensor::take_memory(const Shape& shape, void* data, datatype dtype, layout layout)
	{
		Tensor tensor;

		tensor._desc_ = std::make_unique<neural_primitives::Descriptor>(shape, dtype, layout);
		tensor._memr_ = std::make_unique<neural_primitives::Memory>(*tensor._desc_, data);

		return tensor;
	}

	Tensor Tensor::Random(const Shape& shape, cumeric_t min, cumeric_t max, datatype dtype, layout layout)
	{
		Tensor tensor;
		tensor._desc_ = std::make_unique<neural_primitives::Descriptor>(shape, dtype, layout);
		dim_t count = tensor.lenght();

		float* temp = sycl::malloc_shared<float>(count, internal::device(), internal::sycl_context());

		float f_min = static_cast<float>(min);
		float f_max = static_cast<float>(max);

		oneapi::mkl::rng::philox4x32x10 engine(internal::queue(), 2137);
		oneapi::mkl::rng::uniform<float> dist(f_min, f_max);
		sycl::event rng_event = oneapi::mkl::rng::generate(dist, engine, count, temp);

		sycl::event kernel_event;

		tensor._memr_ = std::make_unique<neural_primitives::Memory>(*tensor._desc_);
		void* data = tensor.data();

		dispatch_datatype(tensor.type(), [&]<typename T>(){
			kernel_event = internal::queue().submit([&](sycl::handler& cgh) {
				cgh.depends_on(rng_event);
				cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
					static_cast<T*>(data)[idx] = static_cast<T>(temp[idx]);
				});
			});
		});

		kernel_event.wait();

		sycl::free(temp, internal::sycl_context());

		return tensor;
	}

	Tensor Tensor::Linspace(cumeric_t start, const cumeric_t end, dim_t num)
	{
		if(num <= 0)
			throw std::invalid_argument("Linspace requires a positive number of elements");

		Tensor tensor(cum::Shape{num}, default_type, layout::X);
		void* values = tensor.data();
		const cumeric_t step = num == 1 ? cumeric_t(0) : (end - start) / static_cast<cumeric_t>(num - 1);


		dispatch_datatype(tensor.type(), [&]<typename T>() -> void
		{
			internal::queue().submit([&](sycl::handler& cgh) -> void
			{
				cgh.parallel_for(sycl::range<1>(num), [=](sycl::id<1> idx) -> void {
					static_cast<T*>(values)[idx] = static_cast<T>(start) + static_cast<cumeric_t>(idx) * step;
				});
			});
		});

		return tensor;
	}

	Tensor Tensor::Zeros(const Shape& shape, datatype dtype, layout layout)
	{
		Tensor tensor(shape, dtype, layout);
		tensor.fill(0);
		return tensor;
	}

	Tensor Tensor::Ones(const Shape& shape, datatype dtype, layout layout)
	{
		Tensor tensor(shape, dtype, layout);
		tensor.fill(1);
		return tensor;
	}

	Tensor Tensor::make_cube(dim_t width, dim_t height, dim_t deepth, datatype dtype, layout layout)
	{
		const auto cube_layout = layout == cum::layout::OI ? cum::layout::ANY : layout;
		return Tensor({deepth, height, width}, dtype, cube_layout);
	}

	Tensor Tensor::make_matrix(dim_t rows, dim_t cols, datatype dtype, layout layout)
	{
		return Tensor({rows, cols}, dtype, layout);
	}

	Tensor Tensor::make_vector(dim_t lenght, datatype dtype, layout layout)
	{
		return Tensor(Shape{lenght}, dtype, layout);
	}

	Tensor Tensor::make_scalar(datatype dtype, layout layout)
	{
		return Tensor(Shape{1}, dtype, layout);
	}


	/**------------------------------------------------------------------------------------------------
	 *                                         Memory
	 *------------------------------------------------------------------------------------------------**/

	Tensor& Tensor::cast(datatype dtype)
	{
		return *this;
	}

	Tensor& Tensor::prefetch()
	{
		memory::prefetch(data(), this->size());
		return *this;
	}

	Tensor& Tensor::to_host()
	{

	}

	Tensor& Tensor::to_device()
	{
		memory::prefetch(data(), this->size());
		return *this;
	}

	/**------------------------------------------------------------------------------------------------
	 *                                         Getters
	 *------------------------------------------------------------------------------------------------**/

	const std::unique_ptr<neural_primitives::Descriptor>& Tensor::descriptor() const
	{
		return _desc_;
	}

	const std::unique_ptr<neural_primitives::Memory>& Tensor::memory() const
	{
		return _memr_;
	}

	dim_t Tensor::size() const
	{
		return _desc_->size();
	}

	dim_t Tensor::rank() const
	{
		return _desc_->ndims();
	}

	dim_t Tensor::dims() const
    {
		return _desc_->ndims();
    }

	dim_t Tensor::lenght() const
    {
		return _desc_->size() / datatype_size(_desc_->type());
    }

	Shape Tensor::shape() const
    {
	    return _desc_->shape();
    }

	layout Tensor::format() const
    {
	    return _desc_->format();
    }

	datatype Tensor::type() const
    {
	    return _desc_->type();
    }

	bool Tensor::has(Axis axis) const
	{
		const Shape tensor_shape = _desc_->shape();
		const int position =        axis_position(rank(), axis);

		return position >= 0 &&
			static_cast<std::size_t>(position) < tensor_shape.size();
	}

	dim_t Tensor::extent(Axis axis) const
	{
		const Shape tensor_shape = _desc_->shape();
		const int position = axis_position(rank(), axis);

		if(position < 0 ||
			static_cast<std::size_t>(position) >= tensor_shape.size())
			return 0;

		return tensor_shape[position];
	}

	dim_t Tensor::batches() const
	{
		return extent(Axis::Batches);
    }

	dim_t Tensor::channels() const
    {
		return extent(Axis::Channels);
    }

	dim_t Tensor::depth() const
    {
		return extent(Axis::Depth);
    }

	dim_t Tensor::height() const
    {
		return extent(Axis::Height);
    }

	dim_t Tensor::width() const
    {
		return extent(Axis::Width);
    }

	dim_t Tensor::rows() const
    {
	    return height();
    }

	dim_t Tensor::cols() const
    {
	    return width();
    }

	std::unique_ptr<neural_primitives::Descriptor>& Tensor::descriptor()
	{
		return _desc_;
	}

	std::unique_ptr<neural_primitives::Memory>& Tensor::memory() // Should this be in accessors section
	{
		return _memr_;
	}

	/**-----------------------------------------------------------------------------------------------------------------------
	 *                                                    Accessors
	 *-----------------------------------------------------------------------------------------------------------------------**/

	dim_t Tensor::compute_index(const Shape& indices) const
	{
		dnnl::memory::desc& md = _desc_->handle().desc;

		const Shape& strides = md.get_strides(); 
		const Shape& shape = this->shape();
	
		dim_t offset = 0;
		for(dim_t _ = 0 ; _ < shape.size() ; _++)
		{
			offset += indices[_] * strides[_];
		}

    	return offset;
	}

	void* Tensor::compute_address(const Shape& indices)
	{
		

	}


	cumeric_t& Tensor::at(const Shape& indices)
	{
		// Tensor element = slice(indices, Shape(rank(), 1));
		//
		// void* element_data = element.data();
		// cumeric_t result;
		// dispatch_datatype(this->type(), [&]<typename T>() -> void
		// {
		// 	result = static_cast<cumeric_t>(static_cast<const T*>(element_data)[0]);
		// });

		return at<cumeric_t>(indices);
	}

	const cumeric_t& Tensor::at(const Shape& indices) const
	{
		// Tensor element = slice(indices, Shape(rank(), 1));
		//
		// void* element_data = element.data();
		// cumeric_t result;
		// dispatch_datatype(this->type(), [&]<typename T>() -> void
		// {
		// 	result = static_cast<cumeric_t>(static_cast<const T*>(element_data)[0]);
		// });

		return at<cumeric_t>(indices);
	}

	// cumeric_t Tensor::at(const Shape& indices) const
	// {
	// 	// dim_t idx = compute_index(indices);
	// 	Tensor element = slice(indices, Shape(rank(), 1));
	//
	// 	void* element_data = element.data();
	// 	cumeric_t result;
	// 	dispatch_datatype(this->type(), [&]<typename T>() -> void
	// 	{
	// 		T value = static_cast<const T*>(element_data)[0];
	// 		result = static_cast<cumeric_t>(value);
	// 	});
	//
	// 	return result;
	// }


	const void* Tensor::data() const
    {
	    return static_cast<void*>(static_cast<std::byte*>(_memr_->data()) + _desc_->offset() * datatype_size(type()));
    }

	void* Tensor::data()
    {
	    return static_cast<void*>(static_cast<std::byte*>(_memr_->data()) + _desc_->offset() * datatype_size(type()));
    }

	cumeric_t Tensor::operator ()(const dim_t row, const dim_t col) const
	{
		return this->at(Shape{row, col});
	}

	cumeric_t Tensor::operator ()(const dim_t idx0, const dim_t idx1, const dim_t idx2) const
	{
		return this->at(Shape{idx0, idx1, idx2});
	}

	cumeric_t Tensor::operator ()(const dim_t idx0, const dim_t idx1, const dim_t idx2, const dim_t idx3) const
	{
		return this->at(Shape{idx0, idx1, idx2, idx3});
	}

	cumeric_t Tensor::operator ()(const dim_t idx0, const dim_t idx1, const dim_t idx2, const dim_t idx3, const dim_t idx4) const
	{
		return this->at(Shape{idx0, idx1, idx2, idx3, idx4});
	}


	Tensor Tensor::batch(dim_t index) const
	{
	}

	Tensor Tensor::channel(dim_t index) const
	{
	}

	Tensor Tensor::row(dim_t index) const
	{
		if (this->rank() != 2)
			throw std::runtime_error("Tensor::row(): rank != 2");

		return slice({index, 0}, {1, cols()});
	}

	Tensor Tensor::col(dim_t index) const
	{
		if (this->rank() != 2)
			throw std::runtime_error("Tensor::col(): rank != 2");

		return slice({0, index}, {rows(), 1});

		// return select_axis(*this, Axis::Cols, index);
	}

	/**------------------------------------------------------------------------------------------------
	*                                         Reshaping
	*------------------------------------------------------------------------------------------------**/

	Tensor Tensor::reshape(const Shape& shape) const
	{

	}

	Tensor& Tensor::reshape_in_place(const Shape& shape)
	{
		return *this;
	}


	Tensor::Tensor(neural_primitives::Descriptor&& desc, const neural_primitives::Memory& source)
		: _desc_(std::make_unique<neural_primitives::Descriptor>(std::move(desc))),
		  _memr_(std::make_unique<neural_primitives::Memory>(*_desc_, source))
	{

	}

	Tensor Tensor::slice(const Shape& offset, const Shape& shape) const
	{
		if (offset.size() != shape.size())
			throw std::invalid_argument("offset and shape must have the same size");

		if(offset.size() != this->rank())
			throw std::invalid_argument("offset and shape must have the same rank");

		for(dim_t i = 0; i < shape.size(); i++)
			if(offset[i] + shape[i] > this->shape()[i])
				throw std::invalid_argument("offset + shape > tensor size");

		const dnnl::memory::desc& md = _desc_->handle().desc;

		const dnnl::memory::desc sub_md= md.submemory_desc(shape, offset);

		return Tensor(neural_primitives::Descriptor({sub_md}), *_memr_);
    }


	Tensor Tensor::transpose()
	{
		Tensor result = *this;

		result.transpose_in_place();
		return result;
    }

	Tensor& Tensor::transpose_in_place()
    {
		if(dims() != 2)
			throw std::invalid_argument("transpose requires a two-dimensional tensor");

		layout format = this->format() == layout::IO ? layout::OI : layout::IO;


		neural_primitives::Descriptor desc(this->shape(), this->type(), format);
		// neural_primitives::Memory mem(desc);
		dnnl::reorder::primitive_desc primitive_desc(
			internal::engine(),
			_desc_->handle().desc,
			internal::engine(),
			desc.handle().desc
		);

		dnnl::reorder(primitive_desc).execute(
			internal::stream(),
	{
				{DNNL_ARG_SRC, _memr_->handle().memory},
				{DNNL_ARG_DST, _memr_->handle().memory}
			}
		);

		internal::stream().wait();


		// _desc_ = std::make_unique<neural_primitives::Descriptor>(desc);

		Shape new_shape = _desc_->shape();
		std::swap(new_shape[0], new_shape[1]);

		_desc_->handle().desc = _desc_->handle().desc.reshape(new_shape); // Is it safe? probably not
		return *this;
    }

	/**------------------------------------------------------------------------------------------------
	 *                                         Reductions
	 *------------------------------------------------------------------------------------------------**/

	cumeric_t Tensor::sum()
    {
		cumeric_t* sum_buff = sycl::malloc_shared<cumeric_t>(1, internal::device(), internal::sycl_context());
    	sum_buff[0] = 0;

    	dnnl::memory::desc sum_desc {
    		Shape(this->dims(), 1), dnnl_data_type(default_type), dnnl_format_tag(format())
    	};

    	dnnl::memory sum_memory = dnnl::sycl_interop::make_memory(
    		sum_desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, sum_buff
    	);

    	dnnl::reduction::primitive_desc pd(
			internal::engine(),
			dnnl::algorithm::reduction_sum,
			_desc_->handle().desc,
			sum_desc,
			0.0f,   // p
			0.0f  // eps
		);

    	dnnl::reduction(pd).execute(
			internal::stream(),
			{
				{DNNL_ARG_SRC, _memr_->handle().memory},
				{DNNL_ARG_DST, sum_memory}
			}
		);

    	internal::stream().wait();

    	cumeric_t result = sum_buff[0];
    	memory::free(sum_buff);

    	return result;
    }

	cumeric_t Tensor::mean()
	{
		return this->sum() / this->lenght();
	}

	cumeric_t Tensor::amean()
	{

	}


	cumeric_t Tensor::squaredNorm()
	{
		return this->squared_norm();
	}

	cumeric_t Tensor::squared_norm()
	{
		cumeric_t* sum_buff = sycl::malloc_shared<cumeric_t>(1, internal::device(), internal::sycl_context());
		sum_buff[0] = 0;

		dnnl::memory::desc sum_desc {
			Shape(this->dims(), 1), dnnl_data_type(default_type), dnnl_format_tag(format())
		};

		dnnl::memory sum_memory = dnnl::sycl_interop::make_memory(
			sum_desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, sum_buff
		);

		// Internal cache has 2048 * 2048 * cumeric_t size bytes
		Tensor square_tensor = take_memory(this->shape(), internal::cache(), this->type(), this->format());

		dnnl::eltwise_forward::primitive_desc square_desc(
			internal::engine(),
			dnnl::prop_kind::forward,
			dnnl::algorithm::eltwise_square,
			this->descriptor()->handle().desc,
			square_tensor.descriptor()->handle().desc
		);

		dnnl::eltwise_forward(square_desc).execute(
			internal::stream(),
			{
				{ DNNL_ARG_SRC, _memr_->handle().memory },
				{ DNNL_ARG_DST, square_tensor._memr_->handle().memory }
			}
		);

		dnnl::reduction::primitive_desc reduction_desc(
			internal::engine(),
			dnnl::algorithm::reduction_sum,
			square_tensor.descriptor()->handle().desc,
			sum_desc,
			0.0f,   // p
			0.0f  // eps
		);

		dnnl::reduction(reduction_desc).execute(
			internal::stream(),
			{
				{DNNL_ARG_SRC, square_tensor.memory()->handle().memory},
				{DNNL_ARG_DST, sum_memory}
			}
		);

		internal::stream().wait();

		cumeric_t result = sum_buff[0];
		memory::free(sum_buff);

		return result;
	}

	Tensor Tensor::colwise_sum()
	{
		if(dims() != 2)
			throw std::invalid_argument("colwise_sum requires a two-dimensional tensor");

		const Shape source_shape = shape();
		Tensor result({source_shape[0], 1}, type(), layout::IO);
		dnnl::reduction::primitive_desc primitive_desc(
			internal::engine(),
			dnnl::algorithm::reduction_sum,
			_desc_->handle().desc,
			result._desc_->handle().desc,
			0.0f,
			0.0f
		);

		dnnl::reduction(primitive_desc).execute(
			internal::stream(),
			{
				{DNNL_ARG_SRC, _memr_->handle().memory},
				{DNNL_ARG_DST, result._memr_->handle().memory}
			}
		);
		internal::stream().wait();
		return result;
	}

	Tensor Tensor::rowwise_sum()
	{
		if(dims() != 2)
			throw std::invalid_argument("rowwise_sum requires a two-dimensional tensor");

		Shape new_shape = shape();
		new_shape[1] = 1;

		Tensor result(new_shape, type(), format());

		dnnl::reduction::primitive_desc primitive_desc(
			internal::engine(),
			dnnl::algorithm::reduction_sum,
			_desc_->handle().desc,
			result._desc_->handle().desc,
			0.0f,
			0.0f
		);

		dnnl::reduction(primitive_desc).execute(
			internal::stream(),
	{
				{DNNL_ARG_SRC, _memr_->handle().memory},
				{DNNL_ARG_DST, result._memr_->handle().memory}
			}
		);
		internal::stream().wait();
		return result;
	}

	Tensor Tensor::channelwise_sum()
	{
		if(dims() != 3)
			throw std::invalid_argument("channelwise_sum requires a tree-dimensional tensor");

		const Shape source_shape = shape();
		Tensor result({1, source_shape[0]}, type(), layout::IO);
		dnnl::reduction::primitive_desc primitive_desc(
			internal::engine(),
			dnnl::algorithm::reduction_sum,
			_desc_->handle().desc,
			result._desc_->handle().desc,
			0.0f,
			0.0f
		);

		dnnl::reduction(primitive_desc).execute(
			internal::stream(),
			{
				{DNNL_ARG_SRC, _memr_->handle().memory},
				{DNNL_ARG_DST, result._memr_->handle().memory}
			}
		);
		internal::stream().wait();
		return result;
	}

	/**------------------------------------------------------------------------------------------------
	 *                                         Elementwise
	 *------------------------------------------------------------------------------------------------**/

	Tensor Tensor::elementwise(functions::function_id function) const
	{
		Tensor result = Tensor(shape(), type(), format());
		neural_primitives::eltwise(result._memr_->handle(), result._desc_->handle(), function, neural_primitives::prop_kind::forward);
		return result;
	}


	Tensor& Tensor::elementwise_in_place(functions::function_id function)
	{
		neural_primitives::eltwise(_memr_->handle(), _desc_->handle(), function, neural_primitives::prop_kind::forward);
		return *this;
	}

	Tensor Tensor::elementwise_diff(functions::function_id function) const
	{
		Tensor result = Tensor(shape(), type(), format());
		neural_primitives::eltwise_diff(result._memr_->handle(), result._desc_->handle(), function, neural_primitives::prop_kind::backward);
		return result;
	}

	Tensor& Tensor::elementwise_diff_in_place(functions::function_id function)
	{
		neural_primitives::eltwise_diff(_memr_->handle(), _desc_->handle(), function, neural_primitives::prop_kind::forward);
		return *this;
	}

	Tensor& Tensor::fill(cumeric_t scalar)
	{
    	void* ptr = static_cast<cumeric_t*>(
			_memr_->handle().memory.get_data_handle()
		);

    	const dim_t count = _desc_->handle().desc.get_size() / sizeof(cumeric_t);

		// functions::various::fill(ptr, scalar, count);

		dispatch_datatype(type(), [&]<typename T>()
		{
			internal::queue().fill(
				static_cast<T*>(ptr),
				static_cast<T>(scalar),
				lenght()
			).wait();
		});

    	runtime::sync();

    	return *this;
	}

	Tensor Tensor::multiply(const Tensor& tensor) const
    {
    	Tensor C(shape(), type(), format());

    	dnnl::binary::primitive_desc primitive_desc {
    		internal::engine(),
			dnnl::algorithm::binary_mul,
			_desc_->handle().desc,
			tensor._desc_->handle().desc,
			C._desc_->handle().desc,
		};

    	dnnl::binary(primitive_desc).execute(
			internal::stream(),
			{
				{ DNNL_ARG_SRC_0, _memr_->handle().memory },
				{ DNNL_ARG_SRC_1, tensor._memr_->handle().memory },
				{ DNNL_ARG_DST, C._memr_->handle().memory }
			}
		);

    	internal::stream().wait();
    	return C;
    }

	Tensor& Tensor::scale(cumeric_t scalar)
	{
		*this *= scalar;

		return *this;
	}

	Tensor Tensor::cwiseProduct(const Tensor& tensor) const
	{
		return this->multiply(tensor);
	}

	Tensor& Tensor::cwise_product_in_place(const Tensor& other)
	{

		dnnl::binary::primitive_desc primitive_desc {
			internal::engine(),
			dnnl::algorithm::binary_mul,
			_desc_->handle().desc,
			other._desc_->handle().desc,
			_desc_->handle().desc,
		};

		dnnl::binary(primitive_desc).execute(
			internal::stream(),
			{
				{ DNNL_ARG_SRC_0, _memr_->handle().memory },
				{ DNNL_ARG_SRC_1, other._memr_->handle().memory },
				{ DNNL_ARG_DST, _memr_->handle().memory }
			}
		);

		internal::stream().wait();
		return *this;
	}

	Tensor Tensor::sqrt() const
	{
		Tensor result = *this;

    	neural_primitives::sqrt(result._memr_->handle(), _memr_->handle(), result._desc_->handle(), _desc_->handle());

    	return result;
	}

	Tensor& Tensor::sqrt_in_place()
	{
    	neural_primitives::sqrt(_memr_->handle(), _desc_->handle());
    	return *this;
	}

	Tensor Tensor::square()
	{
		Tensor result = *this;

    	neural_primitives::square(result._memr_->handle(), _memr_->handle(), result._desc_->handle(), _desc_->handle());

    	return result;
	}

	Tensor& Tensor::square_in_place()
	{
    	neural_primitives::square(_memr_->handle(), _desc_->handle());
    	return *this;
	}

	/**------------------------------------------------------------------------------------------------
	 *                                        Matrix multiplication
	 *------------------------------------------------------------------------------------------------**/

	Tensor Tensor::matmul(const Tensor& other) const
	{
		if (this->cols() != other.rows())
		{
			throw std::invalid_argument("A cols != B rows");
		}

		Shape result_shape = {};
		switch (this->format()) // Temporary solution
		{
			case layout::BA:
			case layout::AB:
			{
				result_shape = {this->rows(), other.cols()};
				break;
			}
		default:
			{
				throw std::invalid_argument("Unsupported format for tensor matmul");
			}
		}

		Tensor C(result_shape, type(), format());

		neural_primitives::matmul(C, *this, other);

		return C;
	}

	/**------------------------------------------------------------------------------------------------
	 *                                         Operator overloads
	 *------------------------------------------------------------------------------------------------**/

	Tensor operator + (const Tensor& A, const Tensor& B)
    {
    	Tensor C(A.shape(), A.type(), A.format());

	    neural_primitives::add(C, A, B);

    	internal::stream().wait();
    	return C;
    }

	Tensor operator - (const Tensor& A, const Tensor& B)
    {
    	Tensor C(A.shape(), A.type(), A.format());

		neural_primitives::sub(C, A, B);

    	internal::stream().wait();
    	return C;
    }

	Tensor operator * (const Tensor& A, const Tensor& B)
    {
		return A.matmul(B);
    }

	Tensor operator / (const Tensor& A, const Tensor& B)
    {
    	Tensor C(A.shape(), A.type(), A.format());

	    dnnl::binary::primitive_desc primitive_desc {
			internal::engine(),
	    	dnnl::algorithm::binary_div,
	    	A._desc_->handle().desc,
	    	B._desc_->handle().desc,
	    	C._desc_->handle().desc,
	    };

    	dnnl::binary(primitive_desc).execute(
    		internal::stream(),
    		{
				{ DNNL_ARG_SRC_0, A._memr_->handle().memory },
				{ DNNL_ARG_SRC_1, B._memr_->handle().memory },
				{ DNNL_ARG_DST, C._memr_->handle().memory }
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
    		Shape(tensor.dims(), 1), dnnl_data_type(default_type),
			tensor.dims() == 1 ? dnnl::memory::format_tag::x : dnnl::memory::format_tag::any
    	};

    	dnnl::memory scalar_memory = dnnl::sycl_interop::make_memory(
    		scalar_desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, scalar_buff
    	);

    	dnnl::binary::primitive_desc primitive_desc {
    		internal::engine(),
			dnnl::algorithm::binary_add,
			tensor._desc_->handle().desc,
			scalar_desc,
			C._desc_->handle().desc,
		};

    	dnnl::binary(primitive_desc).execute(
			internal::stream(),
			{
				{ DNNL_ARG_SRC_0, tensor._memr_->handle().memory },
				{ DNNL_ARG_SRC_1, scalar_memory },
				{ DNNL_ARG_DST, C._memr_->handle().memory }
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
    		Shape(tensor.dims(), 1), dnnl_data_type(default_type),
			tensor.dims() == 1 ? dnnl::memory::format_tag::x : dnnl::memory::format_tag::any
    	};

    	dnnl::memory scalar_memory = dnnl::sycl_interop::make_memory(
    		scalar_desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, scalar_buff
    	);

    	dnnl::binary::primitive_desc primitive_desc {
    		internal::engine(),
			dnnl::algorithm::binary_sub,
			tensor._desc_->handle().desc,
			scalar_desc,
			C._desc_->handle().desc,
		};

    	dnnl::binary(primitive_desc).execute(
			internal::stream(),
			{
				{ DNNL_ARG_SRC_0, tensor._memr_->handle().memory },
				{ DNNL_ARG_SRC_1, scalar_memory },
				{ DNNL_ARG_DST, C._memr_->handle().memory }
			}
		);

    	internal::stream().wait();
    	memory::free(scalar_buff);

    	return C;
    }

	Tensor operator - (const cumeric_t scalar, const Tensor& tensor)
    {
    	Tensor sex(tensor.shape(), tensor.type(), tensor.format());

    	sex.fill(scalar);

    	return sex - tensor;
    }


	Tensor operator * (const Tensor& tensor, cumeric_t scalar)
    {
    	Tensor C(tensor.shape(), tensor.type(), tensor.format());

    	cumeric_t* scalar_buff = memory::allocate(1);
    	scalar_buff[0] = scalar;

    	dnnl::memory::desc scalar_desc {
    		Shape(tensor.dims(), 1), dnnl_data_type(default_type),
			tensor.dims() == 1 ? dnnl::memory::format_tag::x : dnnl::memory::format_tag::any
    	};

    	dnnl::memory scalar_memory = dnnl::sycl_interop::make_memory(
    		scalar_desc, internal::engine(), dnnl::sycl_interop::memory_kind::usm, scalar_buff
    	);

    	dnnl::binary::primitive_desc primitive_desc {
    		internal::engine(),
			dnnl::algorithm::binary_mul,
			tensor._desc_->handle().desc,
			scalar_desc,
			C._desc_->handle().desc,
		};

    	dnnl::binary(primitive_desc).execute(
			internal::stream(),
			{
				{ DNNL_ARG_SRC_0, tensor._memr_->handle().memory },
				{ DNNL_ARG_SRC_1, scalar_memory },
				{ DNNL_ARG_DST, C._memr_->handle().memory }
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
    	Tensor result(tensor.shape(), tensor.type(), tensor.format());

    	cumeric_t* scalar_buff = memory::allocate(1);
    	scalar_buff[0] = scalar;

		neural_primitives::Descriptor scalar_descriptor(Shape(tensor.rank(), 1), tensor.type(), tensor.format());

		neural_primitives::Memory scalar_memory(scalar_descriptor, scalar_buff);

		neural_primitives::div(
			*result.memory(),
			*tensor.memory(),
			scalar_memory,
			*result.descriptor(),
			*tensor.descriptor(),
			scalar_descriptor
		);

    	memory::free(scalar_buff);

    	return result;
    }

	Tensor operator / (const cumeric_t scalar, const Tensor& tensor)
    {
    	Tensor sex(tensor.shape(), tensor.type(), tensor.format());

    	sex.fill(scalar);

    	return sex / tensor;
    }

	Tensor& Tensor::operator += (const Tensor& other)
    {
	    neural_primitives::add(*_memr_, *other._memr_, *_desc_, *other._desc_);
        return *this;
    }

	Tensor& Tensor::operator -= (const Tensor& other)
    {
	    neural_primitives::sub(*_memr_, *other._memr_, *_desc_, *other._desc_);
        return *this;
    }

	Tensor& Tensor::operator *= (const Tensor& other)
    {
	    neural_primitives::mul(*_memr_, *other._memr_, *_desc_, *other._desc_);
        return *this;
    }

	Tensor& Tensor::operator /= (const Tensor& other)
    {
	    neural_primitives::div(*_memr_, *other._memr_, *_desc_, *other._desc_);
        return *this;
    }

	Tensor& Tensor::operator = (const Tensor& other)
    {
		std::println("Assignment operator called");
    	if (this == &other)
    		return *this;

    	auto desc = std::make_unique<neural_primitives::Descriptor>(other.shape(), other.type(), other.format());
    	auto memr = std::make_unique<neural_primitives::Memory>(*_desc_, data());

    	internal::queue().memcpy(memr->data(), other.data(), other.lenght() * datatype_size(this->type())).wait();

		_desc_ = std::move(desc);
		_memr_ = std::move(memr);

    	return *this;
    }

	Tensor& Tensor::operator = (Tensor&&) noexcept = default;
	// Tensor& Tensor::operator = (Tensor&& other) noexcept
	// {
	// 	std::println("Move assignment operator called");
	// 	if (this == &other)
	// 		return *this;
	//
	// 	_desc_ = std::move(other._desc_);
	// 	_memr_ = std::move(other._memr_);
	//
	// 	// internal::queue().copy(_memr)
	// 	return *this;
	// }
  //   {
		// _desc_ = std::move
  //   }


} // cum
