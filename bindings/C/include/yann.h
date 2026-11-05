#ifndef YANN_C_API_H
#define YANN_C_API_H

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * YANN C API - Stable Public Interface
 * ============================================================================ */

#define YANN_API_VERSION_MAJOR 0
#define YANN_API_VERSION_MINOR 3

/* ---------------------------  Scalar Types  ------------------------------- */
typedef int64_t yann_dim_t;
typedef float   yann_scalar_t;

/* --------------------------- Device Selection ----------------------------- */
typedef enum yann_device {
    YANN_DEVICE_AUTO = 0,
    YANN_DEVICE_CPU  = 1,
    YANN_DEVICE_GPU  = 2,
} yann_device_t;

/* --------------------------- Status & Errors ------------------------------ */
typedef enum yann_status {
    YANN_OK = 0,
    YANN_ERROR_INVALID_ARGUMENT,
    YANN_ERROR_OUT_OF_MEMORY,
    YANN_ERROR_DEVICE_UNAVAILABLE,
    YANN_ERROR_RUNTIME,
    YANN_ERROR_NOT_IMPLEMENTED,
} yann_status_t;

/**
 * @brief Retrieve the last error message recorded on the calling thread.
 */
const char* yann_get_last_error(void);

/* --------------------------- Opaque Handles ------------------------------- */
typedef struct yann_context_s   yann_context_t;
typedef struct yann_tensor_s    yann_tensor_t;
typedef struct yann_tensor_s    yann_matrix_t; /* Legacy alias for yann_tensor_t */
typedef struct yann_parameter_s yann_parameter_t;
typedef struct yann_model_s     yann_model_t;
typedef struct yann_chain_s     yann_chain_t;
typedef struct yann_optimizer_s yann_optimizer_t;
typedef struct yann_loss_s      yann_loss_t;

/* --------------------------- Context & Runtime ---------------------------- */
yann_status_t yann_create_context(yann_context_t** out_ctx);
yann_status_t yann_create_context_with_device(yann_device_t device, yann_context_t** out_ctx);
void          yann_destroy_context(yann_context_t* ctx);

void          yann_set_verbosity(unsigned char level);

#ifdef __cplusplus
}
#endif

#include "tensor.h"
#include "parameter.h"
#include "models/sequential.h"

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------- Legacy Matrix Aliases ------------------------ */
yann_status_t yann_matrix_create(yann_context_t* ctx, size_t rows, size_t cols, yann_matrix_t** out_tensor);
yann_status_t yann_matrix_create_from_host(yann_context_t* ctx, size_t rows, size_t cols, const float* host_data, yann_matrix_t** out_tensor);
yann_status_t yann_matrix_copy_to_host(const yann_matrix_t* tensor, float* dst, size_t dst_len);
yann_status_t yann_matrix_get_shape(const yann_matrix_t* tensor, size_t* out_rows, size_t* out_cols);
void          yann_matrix_destroy(yann_matrix_t* tensor);

#ifdef __cplusplus
}
#endif

#endif // YANN_C_API_H
