/*
 * @author: jaro
 * @name:   matrix
 * @file:   modules/cum/bindings/C/include/cum/matrix.h
 * @date:   13 September 2026 22:33:57
 */

#ifndef CUM_MATRIX_H
#define CUM_MATRIX_H

#include "cum/core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cum_matrix_t cum_matrix_t;

cum_status_t cum_matrix_create(cum_matrix_t** matrix, dim_t rows, dim_t cols);

cum_status_t cum_matrix_destroy(cum_matrix_t* matrix);

cum_status_t cum_matrix_rows(const cum_matrix_t* matrix, dim_t* rows);
cum_status_t cum_matrix_cols(const cum_matrix_t* matrix, dim_t* cols);

cum_status_t cum_matrix_add(cum_matrix_t* dst, const cum_matrix_t* lhs, const cum_matrix_t* rhs);
cum_status_t cum_matrix_sub(cum_matrix_t* dst, const cum_matrix_t* lhs, const cum_matrix_t* rhs);
cum_status_t cum_matrix_mul(cum_matrix_t* dst, const cum_matrix_t* lhs, const cum_matrix_t* rhs);
cum_status_t cum_matrix_eltwise_mul(cum_matrix_t* dst, const cum_matrix_t* lhs, const cum_matrix_t* rhs);
cum_status_t cum_matrix_div(cum_matrix_t* dst, const cum_matrix_t* lhs, const cum_matrix_t* rhs);

cum_status_t cum_matrix_add_in_place(cum_matrix_t* lhs, const cum_matrix_t* rhs);
cum_status_t cum_matrix_sub_in_place(cum_matrix_t* lhs, const cum_matrix_t* rhs);
cum_status_t cum_matrix_mul_in_place(cum_matrix_t* lhs, const cum_matrix_t* rhs);
cum_status_t cum_matrix_eltwise_mul_in_place(cum_matrix_t* lhs, const cum_matrix_t* rhs);
cum_status_t cum_matrix_div_in_place(cum_matrix_t* lhs, const cum_matrix_t* rhs);

#ifdef __cplusplus
}
#endif

#endif //CUM_MATRIX_H
