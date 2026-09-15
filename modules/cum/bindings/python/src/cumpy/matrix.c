/*
 * @author: jaro
 * @name:   matrix
 * @file:   modules/cum/bindings/python/src/cumpy/functions/matrix.c
 * @date:   15 September 2026 20:52:13
 */

#include <Python.h>

#include <cum/matrix.h>

typedef struct {
    PyObject_HEAD

    cum_matrix_t mat;
} vector;