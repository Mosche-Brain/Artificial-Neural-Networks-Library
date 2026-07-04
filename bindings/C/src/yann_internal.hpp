#pragma once

#include "yann.h"

#include "cum/Matrix.hpp"
#include "Models/Sequential.hpp"

#include <memory>
#include <string>

/* Internal struct definitions (full types) */
struct yann_context_s {
    // Reserved for future use (device, queue, etc.)
};

struct yann_matrix_s {
    cum::Matrix matrix;
};

struct yann_model_s {
    yann::models::Sequential model;
};

/* Thread-local error message */
inline thread_local std::string g_last_error;

inline void yann_set_error(const std::string& msg) {
    g_last_error = msg;
}

inline void yann_set_error(const char* msg) {
    g_last_error = msg ? msg : "Unknown error";
}