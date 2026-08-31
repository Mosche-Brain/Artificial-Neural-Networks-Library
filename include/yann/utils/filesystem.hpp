#pragma once

#include <cum/Matrix.hpp>
#include <cum/Core.hpp>

#include <cstdint>

namespace yann::utils::filesystem
{
	typedef enum : uint8_t
	{
		INT8 	= 0,
		UINT8 	= 1,
		INT16 	= 2,
		UINT16 	= 3,
		INT32	= 4,
		UINT32	= 5,
		FP16	= 6,
		BF16	= 7,
		FP32	= 8,
		FP64 	= 9
	} DTYPE;

	uint8_t dtype_bytes_count(DTYPE dtype);

	struct MatrixHeader
	{
		uint8_t  meta;
		uint8_t  type;
		uint64_t rows;
		uint64_t cols;
		//MatrixHeader(uint8_t format, uint8_t rows, )

		//uint8_t& format() { return data[0]; }
		//uint8_t& rows()   { return data[1]; }
		//uint8_t& rows()	  { return data[2]; }
		//uint8_t& cols()  { return data[3]; }

		//uint8_t data[4];
	};

	void dump_matrix(const cum::Matrix& mat, const char* path);
	cum::Matrix read_matrix(const char* path);

	void dump_buffer(const cum::cumeric_t* buffer, cum::dim_t size, const char* path);
	void read_buffer(cum::cumeric_t* buffer, cum::dim_t size, const char* path);
}
