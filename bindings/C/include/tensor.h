#ifndef YANN_TENSOR_H
#define YANN_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yann_context_s yann_context_t;
typedef struct yann_tensor_s  yann_tensor_t;
typedef enum yann_status      yann_status_t;

/**
 * @brief Create a 2D tensor filled with zeros.
 */
yann_status_t yann_tensor_create(yann_context_t* ctx, size_t rows, size_t cols, yann_tensor_t** out_tensor);

/**
 * @brief Create an N-D tensor filled with zeros.
 */
yann_status_t yann_tensor_create_nd(yann_context_t* ctx, const int64_t* dims, size_t ndims, yann_tensor_t** out_tensor);

/**
 * @brief Create a 2D tensor and populate it from a contiguous float buffer on the host.
 */
yann_status_t yann_tensor_create_from_host(yann_context_t* ctx, size_t rows, size_t cols, const float* host_data, yann_tensor_t** out_tensor);

/**
 * @brief Create an N-D tensor and populate it from a contiguous float buffer on the host.
 */
yann_status_t yann_tensor_create_nd_from_host(yann_context_t* ctx, const int64_t* dims, size_t ndims, const float* host_data, yann_tensor_t** out_tensor);

/**
 * @brief Copy tensor elements to a contiguous host float buffer.
 */
yann_status_t yann_tensor_copy_to_host(const yann_tensor_t* tensor, float* dst, size_t dst_len);

/**
 * @brief Copy elements from a contiguous host float buffer into an existing tensor.
 */
yann_status_t yann_tensor_copy_from_host(yann_tensor_t* tensor, const float* src, size_t src_len);

/**
 * @brief Fill a tensor with a constant scalar value.
 */
yann_status_t yann_tensor_fill(yann_tensor_t* tensor, float value);

/**
 * @brief Get a single scalar element at 2D coordinates (row, col).
 */
yann_status_t yann_tensor_get_2d(const yann_tensor_t* tensor, size_t row, size_t col, float* out_value);

/**
 * @brief Set a single scalar element at 2D coordinates (row, col).
 */
yann_status_t yann_tensor_set_2d(yann_tensor_t* tensor, size_t row, size_t col, float value);

/**
 * @brief Query 2D dimensions (rows, cols) of a tensor.
 */
yann_status_t yann_tensor_get_shape(const yann_tensor_t* tensor, size_t* out_rows, size_t* out_cols);

/**
 * @brief Query N-D rank of a tensor.
 */
size_t yann_tensor_get_rank(const yann_tensor_t* tensor);

/**
 * @brief Query total number of elements in a tensor.
 */
size_t yann_tensor_get_length(const yann_tensor_t* tensor);

/**
 * @brief Copy N-D shape dimensions into caller-provided array.
 */
yann_status_t yann_tensor_get_dims(const yann_tensor_t* tensor, int64_t* out_dims, size_t max_dims);

/**
 * @brief Save a single tensor to a .safetensors file.
 */
yann_status_t yann_tensor_save_safetensors(const yann_tensor_t* tensor, const char* filename, const char* name);

/**
 * @brief Load a single tensor from a .safetensors file.
 */
yann_status_t yann_tensor_load_safetensors(const char* filename, const char* name, yann_tensor_t** out_tensor);

/**
 * @brief Destroy a tensor and release its underlying memory.
 */
void yann_tensor_destroy(yann_tensor_t* tensor);

#ifdef __cplusplus
}
#endif

#endif // YANN_TENSOR_H
