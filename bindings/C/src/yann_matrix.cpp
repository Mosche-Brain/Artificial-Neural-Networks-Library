#include "yann_internal.hpp"
#include <vector>

extern "C" {

static std::vector<cum::cumeric_t> convert_host_to_internal(const float* src, size_t n)
{
    std::vector<cum::cumeric_t> dst(n);
    for (size_t i = 0; i < n; ++i)
        dst[i] = static_cast<cum::cumeric_t>(src[i]);
    return dst;
}

static void convert_internal_to_host(const cum::cumeric_t* src, float* dst, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        dst[i] = static_cast<float>(src[i]);
}

yann_status_t yann_matrix_create(yann_context_t* ctx,
                                 size_t rows, size_t cols,
                                 yann_matrix_t** out_tensor)
{
    if (!out_tensor) {
        yann_set_error("yann_matrix_create: out_tensor is null");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto* t = new yann_matrix_s{};
        t->matrix = cum::Matrix(rows, cols, static_cast<cum::cumeric_t>(0));
        *out_tensor = t;
        return YANN_OK;
    } catch (const std::exception& e) {
        yann_set_error(std::string("yann_matrix_create failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_matrix_create_from_host(yann_context_t* ctx,
                                           size_t rows, size_t cols,
                                           const float* host_data,
                                           yann_matrix_t** out_tensor)
{
    if (!out_tensor || !host_data) {
        yann_set_error("yann_matrix_create_from_host: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto* t = new yann_matrix_s{};
        const size_t total = rows * cols;
        auto internal_data = convert_host_to_internal(host_data, total);
        t->matrix = cum::Matrix(rows, cols, internal_data.data());
        *out_tensor = t;
        return YANN_OK;
    } catch (const std::exception& e) {
        yann_set_error(std::string("yann_matrix_create_from_host failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_matrix_copy_to_host(yann_matrix_t* tensor, float* dst, size_t dst_len)
{
    if (!tensor || !dst) {
        yann_set_error("yann_matrix_copy_to_host: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try {
        const size_t total = tensor->matrix.rows() * tensor->matrix.cols();
        if (dst_len < total) {
            yann_set_error("yann_matrix_copy_to_host: destination buffer too small");
            return YANN_ERROR_INVALID_ARGUMENT;
        }
        convert_internal_to_host(tensor->matrix.data(), dst, total);
        return YANN_OK;
    } catch (const std::exception& e) {
        yann_set_error(std::string("yann_matrix_copy_to_host failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_matrix_get_shape(yann_matrix_t* tensor, size_t* out_rows, size_t* out_cols)
{
    if (!tensor || !out_rows || !out_cols) {
        yann_set_error("yann_matrix_get_shape: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }
    *out_rows = tensor->matrix.rows();
    *out_cols = tensor->matrix.cols();
    return YANN_OK;
}

void yann_matrix_destroy(yann_matrix_t* tensor)
{
    delete tensor;
}

} // extern "C"