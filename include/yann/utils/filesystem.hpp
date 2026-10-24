/**
 * @file filesystem.hpp
 * @brief Binary file I/O operations for matrices, memory buffers, and Sequential models.
 */

#pragma once

#include "Sequential.hpp"
#include <cum/Matrix.hpp>
#include <cum/Core.hpp>

#include <cstdint>

namespace yann::utils::filesystem
{
    /**
     * @enum DTYPE
     * @brief Serialization data type identifiers.
     */
	typedef enum : uint8_t
	{
		INT8 	= 0,  /**< Signed 8-bit integer. */
		UINT8 	= 1,  /**< Unsigned 8-bit integer. */
		INT16 	= 2,  /**< Signed 16-bit integer. */
		UINT16 	= 3,  /**< Unsigned 16-bit integer. */
		INT32	= 4,  /**< Signed 32-bit integer. */
		UINT32	= 5,  /**< Unsigned 32-bit integer. */
		FP16	= 6,  /**< 16-bit half-precision float. */
		BF16	= 7,  /**< 16-bit brain floating point. */
		FP32	= 8,  /**< 32-bit single-precision float. */
		FP64 	= 9   /**< 64-bit double-precision float. */
	} DTYPE;

    /**
     * @brief Returns the number of bytes required to store an element of the specified DTYPE.
     * @param dtype Data type enum value.
     * @return Byte size of the data type.
     */
	uint8_t dtype_bytes_count(DTYPE dtype);

    /**
     * @struct MatrixHeader
     * @brief Binary file header preceding serialized matrix payloads.
     */
	struct MatrixHeader
	{
		uint8_t  meta; /**< Metadata flags and layout info. */
		uint8_t  type; /**< Data type identifier (DTYPE). */
		uint64_t rows; /**< Number of matrix rows. */
		uint64_t cols; /**< Number of matrix columns. */
		//MatrixHeader(uint8_t format, uint8_t rows, )

		//uint8_t& format() { return data[0]; }
		//uint8_t& rows()   { return data[1]; }
		//uint8_t& rows()	  { return data[2]; }
		//uint8_t& cols()  { return data[3]; }

		//uint8_t data[4];
	};

    /**
     * @brief Dump a matrix to a binary file on disk.
     * @param mat Matrix to serialize.
     * @param path File system destination path.
     */
	void dump_matrix(const cum::Matrix& mat, const char* path);

    /**
     * @brief Read a matrix previously serialized with dump_matrix from disk.
     * @param path Path to the binary matrix file.
     * @return Deserialized cum::Matrix.
     */
	cum::Matrix read_matrix(const char* path);

    /**
     * @brief Write a raw memory buffer to a file.
     * @param buffer Pointer to contiguous memory buffer.
     * @param size Number of elements to write.
     * @param path File system destination path.
     */
	void dump_buffer(const cum::cumeric_t* buffer, cum::dim_t size, const char* path);

    /**
     * @brief Read raw memory buffer contents from a file.
     * @param buffer Pointer to destination memory buffer.
     * @param size Number of elements to read.
     * @param path File system source path.
     */
	void read_buffer(cum::cumeric_t* buffer, cum::dim_t size, const char* path);

    /**
     * @brief Dump a Sequential model's parameters and weights to disk.
     * @param model Sequential model to serialize.
     * @param path Destination file path.
     */
	void dump_model(models::Sequential& model, const char* path);

    /**
     * @brief Load model parameters and weights into an existing Sequential model.
     * @param model Target Sequential model to populate.
     * @param path Source file path.
     */
	void load_model(models::Sequential& model, const char* path);
}
