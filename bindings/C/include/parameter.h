#ifndef YANN_PARAMETER_H
#define YANN_PARAMETER_H

#include <stdbool.h>
#include <stddef.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct yann_parameter_s yann_parameter_t;

yann_status_t yann_parameter_create_zeros(size_t rows, size_t cols, yann_parameter_t** out_param);
yann_status_t yann_parameter_create_ones(size_t rows, size_t cols, yann_parameter_t** out_param);
yann_status_t yann_parameter_create_uniform(size_t rows, size_t cols, yann_parameter_t** out_param);
yann_status_t yann_parameter_create_he(size_t rows, size_t cols, yann_parameter_t** out_param);
yann_status_t yann_parameter_create_xavier(size_t rows, size_t cols, yann_parameter_t** out_param);

yann_status_t yann_parameter_load_safetensors(const char* filename, const char* name, bool load_gradient, yann_parameter_t** out_param);
yann_status_t yann_parameter_save_safetensors(const yann_parameter_t* param, const char* filename, const char* name, bool include_gradient);

yann_status_t yann_parameter_get_values(const yann_parameter_t* param, yann_tensor_t** out_tensor);
yann_status_t yann_parameter_get_gradient(const yann_parameter_t* param, yann_tensor_t** out_tensor);
yann_status_t yann_parameter_clear_gradient(yann_parameter_t* param);

void yann_parameter_destroy(yann_parameter_t* param);

#ifdef __cplusplus
}
#endif

#endif // YANN_PARAMETER_H
