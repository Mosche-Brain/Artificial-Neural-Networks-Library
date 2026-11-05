#ifndef YANN_SEQUENTIAL_H
#define YANN_SEQUENTIAL_H

#include <stddef.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yann_model_s     yann_model_t;
typedef struct yann_chain_s     yann_chain_t;
typedef struct yann_optimizer_s yann_optimizer_t;
typedef struct yann_loss_s      yann_loss_t;

/* --------------------------- Optimizers ----------------------------------- */
yann_status_t yann_optimizer_create_sgd(float learning_rate, yann_optimizer_t** out_opt);
yann_status_t yann_optimizer_create_adam(float learning_rate, float b1, float b2, yann_optimizer_t** out_opt);
yann_status_t yann_optimizer_create_momentum(float learning_rate, float beta, yann_optimizer_t** out_opt);
yann_status_t yann_optimizer_create_adagrad(float learning_rate, float eps, yann_optimizer_t** out_opt);
void          yann_optimizer_destroy(yann_optimizer_t* opt);

/* --------------------------- Loss Functions ------------------------------- */
yann_status_t yann_loss_create_mse(yann_loss_t** out_loss);
yann_status_t yann_loss_create_mae(yann_loss_t** out_loss);
yann_status_t yann_loss_create_bce(yann_loss_t** out_loss);
yann_status_t yann_loss_create_cce(int from_logits, yann_loss_t** out_loss);
yann_status_t yann_loss_compute(yann_loss_t* loss, const yann_tensor_t* predicted, const yann_tensor_t* target, float* out_value);
void          yann_loss_destroy(yann_loss_t* loss);

/* --------------------------- Sequential Model ----------------------------- */
yann_status_t yann_model_create(yann_context_t* ctx, yann_model_t** out_model);
void          yann_model_destroy(yann_model_t* model);

yann_status_t yann_model_add_input(yann_model_t* model, size_t features);
yann_status_t yann_model_add_input_nd(yann_model_t* model, const int64_t* dims, size_t ndims);
yann_status_t yann_model_add_dense(yann_model_t* model, size_t units, const char* activation);
yann_status_t yann_model_add_linear(yann_model_t* model, size_t units);
yann_status_t yann_model_add_flatten(yann_model_t* model);
yann_status_t yann_model_add_softmax(yann_model_t* model, size_t length, int64_t axis);

yann_status_t yann_model_build(yann_model_t* model, const int64_t* input_dims, size_t ndims);

yann_status_t yann_model_forward(yann_model_t* model, const yann_tensor_t* input, yann_tensor_t** output);
yann_status_t yann_model_predict(yann_model_t* model, const yann_tensor_t* input, yann_tensor_t** output);
yann_status_t yann_model_backward(yann_model_t* model, const yann_tensor_t* d_output, yann_tensor_t** d_input);

/**
 * @brief Train the Sequential model with explicit loss, optimizer, epochs, and batch size.
 */
yann_status_t yann_model_fit_ex(
    yann_model_t* model,
    const yann_tensor_t* X,
    const yann_tensor_t* Y,
    yann_loss_t* loss,
    yann_optimizer_t* optimizer,
    size_t epochs,
    size_t batch_size
);

/**
 * @brief Convenience training helper using MSE loss and SGD optimizer with batch_size=1.
 */
yann_status_t yann_model_fit(
    yann_model_t* model,
    const yann_tensor_t* X,
    const yann_tensor_t* Y,
    float learning_rate,
    size_t epochs
);

yann_status_t yann_model_save_safetensors(const yann_model_t* model, const char* filename);
yann_status_t yann_model_load_safetensors(yann_model_t* model, const char* filename);

size_t yann_model_get_layer_count(const yann_model_t* model);

/* --------------------------- Chain View ----------------------------------- */
yann_status_t yann_chain_create(yann_model_t* const* models, size_t model_count, yann_chain_t** out_chain);
yann_status_t yann_chain_add_model(yann_chain_t* chain, yann_model_t* model);
yann_status_t yann_chain_build(yann_chain_t* chain, const int64_t* input_dims, size_t ndims);
yann_status_t yann_chain_forward(yann_chain_t* chain, const yann_tensor_t* input, yann_tensor_t** output);
yann_status_t yann_chain_backward(yann_chain_t* chain, const yann_tensor_t* d_output, yann_tensor_t** d_input);
yann_status_t yann_chain_fit(
    yann_chain_t* chain,
    const yann_tensor_t* X,
    const yann_tensor_t* Y,
    yann_loss_t* loss,
    yann_optimizer_t* optimizer,
    size_t epochs,
    size_t batch_size
);
size_t yann_chain_get_model_count(const yann_chain_t* chain);
void   yann_chain_destroy(yann_chain_t* chain);

#ifdef __cplusplus
}
#endif

#endif // YANN_SEQUENTIAL_H
