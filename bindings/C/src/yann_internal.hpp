#pragma once

#include "yann.h"

#include "cum/Matrix.hpp"
#include "Models/Sequential.hpp"

#include <memory>
#include <string>

/* ============================================================================
 * Internal structures for the YANN C API
 *
 * These are NOT part of the public API and can change at any time.
 * ============================================================================ */

struct yann_context_t
{
    // Future: can hold SYCL queue, device selection, memory pool, etc.
    // For now it mostly serves as a lifetime anchor.
};

struct yann_tensor_t
{
    // Always stores data using the library's internal cumeric_t type.
    // The public C API always speaks in float (FP32).
    cum::Matrix matrix;
};

struct yann_model_t
{
    YANN::Models::Sequential model;
};

/* --------------------------- Error Handling ------------------------------- */

inline thread_local std::string g_last_error;

inline void yann_set_error(const std::string& msg)
{
    g_last_error = msg;
}

inline void yann_set_error(const char* msg)
{
    g_last_error = msg ? msg : "Unknown error";
}
