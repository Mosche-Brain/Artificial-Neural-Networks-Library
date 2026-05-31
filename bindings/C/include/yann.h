#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * YANN C API - Stable Public Interface
 *
 * This API is designed to be resilient to internal C++ changes.
 * All implementation details are hidden behind opaque handles.
 *
 * Data Type Policy:
 *   - The public C API uses 'float' (FP32) for all tensor data exchange.
 *   - The library may be compiled with different internal precision
 *     (FP64, FP32, FP16, BF16, int8) via cumeric_t.
 *   - Conversion happens internally. This guarantees ABI stability.
 * ============================================================================ */

/* --------------------------- Versioning ----------------------------------- */
#define YANN_API_VERSION_MAJOR 0
#define YANN_API_VERSION_MINOR 1
#define YANN_API_VERSION_PATCH 0

/* --------------------------- Opaque Handles ------------------------------- */
typedef struct yann_context_t*   yann_context_t;
typedef struct yann_tensor_t*    yann_tensor_t;
typedef struct yann_model_t*     yann_model_t;

/* --------------------------- Status & Errors ------------------------------ */
typedef enum yann_status {
    YANN_OK = 0,
    YANN_ERROR_INVALID_ARGUMENT,
    YANN_ERROR_OUT_OF_MEMORY,
    YANN_ERROR_DEVICE_UNAVAILABLE,
    YANN_ERROR_RUNTIME,
    YANN_ERROR_NOT_IMPLEMENTED,
    YANN_ERROR_MODEL_NOT_COMPILED,
} yann_status_t;

/**
 * Returns a human-readable description of the last error.
 * The returned string is valid until the next YANN C API call on the same thread.
 */
const char* yann_get_last_error(void);

/* --------------------------- Context -------------------------------------- */
/**
 * Creates a new YANN context. The context owns device resources (SYCL queue, etc.).
 * Must be destroyed with yann_destroy_context().
 */
yann_status_t yann_create_context(yann_context_t* out_ctx);

/**
 * Destroys a context and releases all associated resources.
 */
void yann_destroy_context(yann_context_t ctx);

/* --------------------------- Tensor --------------------------------------- */
/**
 * Creates an uninitialized tensor of shape [rows, cols].
 * Data is allocated on the device (or unified memory depending on build).
 */
yann_status_t yann_tensor_create(yann_context_t ctx,
                                 size_t rows, size_t cols,
                                 yann_tensor_t* out_tensor);

/**
 * Creates a tensor and initializes it from host memory (row-major float32).
 * The data is copied into the library's memory space.
 */
yann_status_t yann_tensor_create_from_host(yann_context_t ctx,
                                           size_t rows, size_t cols,
                                           const float* host_data,
                                           yann_tensor_t* out_tensor);

/**
 * Copies tensor data back to host memory (must be pre-allocated, row-major float32).
 */
yann_status_t yann_tensor_copy_to_host(yann_tensor_t tensor,
                                       float* dst,
                                       size_t dst_len);

/**
 * Returns the shape of the tensor.
 */
yann_status_t yann_tensor_get_shape(yann_tensor_t tensor,
                                    size_t* out_rows,
                                    size_t* out_cols);

/**
 * Destroys a tensor and frees its resources.
 */
void yann_tensor_destroy(yann_tensor_t tensor);

/* --------------------------- Model ---------------------------------------- */
/**
 * Creates an empty Sequential-style model.
 */
yann_status_t yann_model_create(yann_context_t ctx, yann_model_t* out_model);

/**
 * Destroys the model and all its layers/weights.
 */
void yann_model_destroy(yann_model_t model);

/**
 * Adds a Dense layer to the model.
 *
 * @param units       Number of neurons in the layer
 * @param activation  Activation function name: "relu", "linear", "sigmoid", "tanh", "softmax"
 */
yann_status_t yann_model_add_dense(yann_model_t model,
                                   size_t units,
                                   const char* activation);

/**
 * (Optional) Explicitly sets the input size. Call before adding layers if needed.
 * Many models can infer input size from the first forward pass.
 */
yann_status_t yann_model_set_input_size(yann_model_t model, size_t input_size);

/**
 * Performs a forward pass (inference).
 *
 * @param input   Input tensor (must match the expected input size)
 * @param output  Output tensor handle. The tensor is created by the callee
 *                and must be destroyed by the caller using yann_tensor_destroy().
 */
yann_status_t yann_model_predict(yann_model_t model,
                                 yann_tensor_t input,
                                 yann_tensor_t* output);

/**
 * Trains the model for a number of epochs using simple gradient descent.
 * This is a convenience function. For more control, use the training API (future).
 */
yann_status_t yann_model_fit(yann_model_t model,
                             yann_tensor_t X,
                             yann_tensor_t Y,
                             float learning_rate,
                             size_t epochs);

/* --------------------------- Introspection -------------------------------- */
/**
 * Returns the number of layers in the model.
 */
size_t yann_model_get_layer_count(yann_model_t model);

/**
 * Returns the output size of a specific layer (0-based).
 */
yann_status_t yann_model_get_layer_output_size(yann_model_t model,
                                               size_t layer_index,
                                               size_t* out_size);

#ifdef __cplusplus
}
#endif
