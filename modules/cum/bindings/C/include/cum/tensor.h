#ifndef CUM_TENSOR_H
#define CUM_TENSOR_H

#include "cum/core.h"

#ifdef __cplusplus
extern "C" {
#endif

cum_status_t cum_tensor_create(
    cum_tensor_t** tensor,
    cum_shape_t shape,
    cum_datatype_t datatype,
    cum_layout_t layout
);
cum_status_t cum_tensor_destroy(cum_tensor_t* tensor);
cum_status_t cum_tensor_copy(cum_tensor_t* dst, const cum_tensor_t* src);

cum_status_t cum_tensor_shape(
    const cum_tensor_t* tensor,
    dim_t* dims,
    size_t capacity,
    size_t* rank
);
cum_status_t cum_tensor_size(const cum_tensor_t* tensor, dim_t* size);
cum_status_t cum_tensor_length(const cum_tensor_t* tensor, dim_t* length);
cum_status_t cum_tensor_datatype(const cum_tensor_t* tensor, cum_datatype_t* datatype);
cum_status_t cum_tensor_layout(const cum_tensor_t* tensor, cum_layout_t* layout);
cum_status_t cum_tensor_data(const cum_tensor_t* tensor, const void** data);

cum_status_t cum_tensor_fill(cum_tensor_t* tensor, cumeric_t scalar);
cum_status_t cum_tensor_add(cum_tensor_t* dst, const cum_tensor_t* lhs, const cum_tensor_t* rhs);
cum_status_t cum_tensor_sub(cum_tensor_t* dst, const cum_tensor_t* lhs, const cum_tensor_t* rhs);
cum_status_t cum_tensor_mul(cum_tensor_t* dst, const cum_tensor_t* lhs, const cum_tensor_t* rhs);
cum_status_t cum_tensor_div(cum_tensor_t* dst, const cum_tensor_t* lhs, const cum_tensor_t* rhs);
cum_status_t cum_tensor_mmul(cum_tensor_t* dst, const cum_tensor_t* lhs, const cum_tensor_t* rhs);

cum_status_t cum_tensor_add_in_place(cum_tensor_t* lhs, const cum_tensor_t* rhs);
cum_status_t cum_tensor_sub_in_place(cum_tensor_t* lhs, const cum_tensor_t* rhs);
cum_status_t cum_tensor_mul_in_place(cum_tensor_t* lhs, const cum_tensor_t* rhs);
cum_status_t cum_tensor_div_in_place(cum_tensor_t* lhs, const cum_tensor_t* rhs);

#ifdef __cplusplus
}
#endif

#endif
