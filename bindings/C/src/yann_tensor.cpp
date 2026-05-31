#include "yann_internal.hpp"

#include <vector>
#include <algorithm>
#include <cstring>

extern "C" {

// Helper: convert float host data -> internal cumeric_t
static std::vector<cum::cumeric_t> convert_host_to_internal(const float* src, size_t n)
{
    std::vector<cum::cumeric_t> dst(n);
    for (size_t i = 0; i < n; ++i)
    {
        dst[i] = static_cast<cum::cumeric_t>(src[i]);
    }
    return dst;
}

// Helper: convert internal cumeric_t -> float host data
static void convert_internal_to_host(const cum::cumeric_t* src, float* dst, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        dst[i] = static_cast<float>(src[i]);
    }
}

yann_status_t yann_tensor_create(yann_context_t ctx,
                                 size_t rows, size_t cols,
                                 yann_tensor_t* out_tensor)
{
    if (!ctx || !out_tensor)
    {
        yann_set_error("yann_tensor_create: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        auto* t = new yann_tensor_t{};
        t->matrix = cum::Matrix(rows, cols, static_cast<cum::cumeric_t>(0));
        *out_tensor = t;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_create failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_tensor_create_from_host(yann_context_t ctx,
                                           size_t rows, size_t cols,
                                           const float* host_data,
                                           yann_tensor_t* out_tensor)
{
    if (!ctx || !out_tensor || !host_data)
    {
        yann_set_error("yann_tensor_create_from_host: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        auto* t = new yann_tensor_t{};
        const size_t total = rows * cols;

        auto internal_data = convert_host_to_internal(host_data, total);

        // Use the constructor that takes a raw pointer (it copies internally)
        t->matrix = cum::Matrix(rows, cols, internal_data.data());
        *out_tensor = t;
        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_create_from_host failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_tensor_copy_to_host(yann_tensor_t tensor, float* dst, size_t dst_len)
{
    if (!tensor || !dst)
    {
        yann_set_error("yann_tensor_copy_to_host: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    try
    {
        const size_t total = tensor->matrix.rows() * tensor->matrix.cols();

        if (dst_len < total)
        {
            yann_set_error("yann_tensor_copy_to_host: destination buffer too small");
            return YANN_ERROR_INVALID_ARGUMENT;
        }

        // Because we use USM (malloc_shared), we can read directly, but we still do a safe copy
        const cum::cumeric_t* src = tensor->matrix.data();
        convert_internal_to_host(src, dst, total);

        return YANN_OK;
    }
    catch (const std::exception& e)
    {
        yann_set_error(std::string("yann_tensor_copy_to_host failed: ") + e.what());
        return YANN_ERROR_RUNTIME;
    }
}

yann_status_t yann_tensor_get_shape(yann_tensor_t tensor, size_t* out_rows, size_t* out_cols)
{
    if (!tensor || !out_rows || !out_cols)
    {
        yann_set_error("yann_tensor_get_shape: invalid argument");
        return YANN_ERROR_INVALID_ARGUMENT;
    }

    *out_rows = tensor->matrix.rows();
    *out_cols = tensor->matrix.cols();
    return YANN_OK;
}

void yann_tensor_destroy(yann_tensor_t tensor)
{
    if (tensor)
    {
        // cum::Matrix destructor will call sycl::free on its data
        delete tensor;
    }
}

} // extern "C"
