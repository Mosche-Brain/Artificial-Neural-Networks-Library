#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * YANN C API - Stable Public Interface
 * ============================================================================ */

#define YANN_API_VERSION_MAJOR 0
#define YANN_API_VERSION_MINOR 2

/* ---------------------------  Scalar Types  ------------------------------- */
typedef long long yann_dim_t;
typedef float     yann_scalar_t;

/* --------------------------- Opaque Handles ------------------------------- */
typedef struct yann_context_s  yann_context_t;
typedef struct yann_matrix_s   yann_matrix_t;
typedef struct yann_vector_s   yann_vector_t;
typedef struct yann_model_s    yann_model_t;

/* --------------------------- Status & Errors ------------------------------ */
typedef enum yann_status {
    YANN_OK = 0,
    YANN_ERROR_INVALID_ARGUMENT,
    YANN_ERROR_OUT_OF_MEMORY,
    YANN_ERROR_DEVICE_UNAVAILABLE,
    YANN_ERROR_RUNTIME,
    YANN_ERROR_NOT_IMPLEMENTED,
} yann_status_t;

const char* yann_get_last_error(void);

/* --------------------------- Context -------------------------------------- */
yann_status_t yann_create_context(yann_context_t** out_ctx);
void          yann_destroy_context(yann_context_t* ctx);

/* --------------------------- Tensor --------------------------------------- */
yann_status_t yann_matrix_create(yann_context_t* ctx,
                                 size_t rows, size_t cols,
                                 yann_matrix_t** out_tensor);

yann_status_t yann_matrix_create_from_host(yann_context_t* ctx,
                                           size_t rows, size_t cols,
                                           const float* host_data,
                                           yann_matrix_t** out_tensor);

yann_status_t yann_matrix_copy_to_host(yann_matrix_t* tensor,
                                       float* dst,
                                       size_t dst_len);

yann_status_t yann_matrix_get_shape(yann_matrix_t* tensor,
                                    size_t* out_rows,
                                    size_t* out_cols);

void yann_matrix_destroy(yann_matrix_t* tensor);

/* --------------------------- Model ---------------------------------------- */
yann_status_t yann_model_create(yann_context_t* ctx, yann_model_t** out_model);
void          yann_model_destroy(yann_model_t* model);

yann_status_t yann_model_add_dense(yann_model_t* model,
                                   size_t units,
                                   const char* activation);

yann_status_t yann_model_predict(yann_model_t* model,
                                 yann_matrix_t* input,
                                 yann_matrix_t** output);

yann_status_t yann_model_fit(yann_model_t* model,
                             yann_matrix_t* X,
                             yann_matrix_t* Y,
                             float learning_rate,
                             size_t epochs);

size_t yann_model_get_layer_count(yann_model_t* model);

#ifdef __cplusplus
}

inline const char* yann_get_last_error()
{
}
#endif
