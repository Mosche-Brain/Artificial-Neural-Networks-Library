/*
 * @author: jaro
 * @name:   tensor
 * @file:   modules/cum/bindings/C/include/cum/tensor.h
 * @date:   13 September 2026 22:34:31
 */

#ifndef YANN_TENSOR_H
#define YANN_TENSOR_H

#include "cum/core.h"

typedef struct cum_tensor_t cum_tensor_t;

typedef struct cum_datatype_t cum_datatype_t;
typedef struct cum_descriptor_t cum_descriptor_t;
typedef struct cum_memory_t cum_memory_t;
typedef struct cum_shape_t cum_shape_t;
typedef struct cum_layout_t cum_layout_t;

__event__ cum_create_tensor(cum_shape_t* shape);

void __

__event__ cum_tensor_mul(cum_tensor_t* C, cum_tensor_t* A, cum_tensor_t* B);
__event__ cum_tensor_add(cum_tensor_t* C, cum_tensor_t* A, cum_tensor_t* B);
__event__ cum_tensor_sub(cum_tensor_t* C, cum_tensor_t* A, cum_tensor_t* B);
__event__ cum_tensor_div(cum_tensor_t* C, cum_tensor_t* A, cum_tensor_t* B);
__event__ cum_tensor_mmul(cum_tensor_t* C, cum_tensor_t* A, cum_tensor_t* B);

__event__ cum_tensor_scale(cum_tensor_t* dst, cum_tensor_t* src, cumeric_t scal);
__event__ cum_tensor_trans(cum_tensor_t* dst, cum_tensor_t* src);

#endif //YANN_TENSOR_H
